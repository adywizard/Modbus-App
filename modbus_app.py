import json
import re
import csv
from pathlib import Path
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ---- Modern pymodbus (>= 3.10): only non-deprecated imports ----
from pymodbus.client import ModbusTcpClient, ModbusSerialClient

# ---- Serial port enumeration ----
try:
    from serial.tools import list_ports
except Exception:
    list_ports = None  # guarded where used


APP_TITLE = "Modbus Reader/Writer -- Made by Adywizard with help of Copilot"
DEFAULT_TIMEOUT = 2.0
SETTINGS_PATH = Path(__file__).resolve().with_name("modbus_settings.json")

# Common baud rates for RTU combobox
BAUD_RATES = [
    "300", "600", "1200", "2400", "4800",
    "9600", "14400", "19200", "28800", "38400",
    "57600", "115200", "230400", "460800", "921600"
]

# Log levels & filtering
LEVEL_ORDER = {"INFO": 1, "WARN": 2, "ERROR": 3}
LOG_FILTER_VALUES = ["All", "INFO", "WARN", "ERROR"]


# ------------- Utility helpers -------------
def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def safe_int(s: str, default: int = 0) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return default


def safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default


def show_error(title: str, err: Exception | str):
    messagebox.showerror(title, str(err))


def ensure_connected(client) -> bool:
    return client is not None


def _format_port_entry(p) -> str:
    """
    Build a friendly label for a serial port combobox entry.
    p is a serial.tools.list_ports_common.ListPortInfo.
    """
    if not p:
        return ""
    label = p.device
    extras = []
    if getattr(p, "description", None):
        extras.append(p.description)
    if getattr(p, "manufacturer", None):
        extras.append(p.manufacturer)
    vid = getattr(p, "vid", None)
    pid = getattr(p, "pid", None)
    if vid is not None and pid is not None:
        extras.append(f"VID:PID={vid:04X}:{pid:04X}")
    if extras:
        label += " – " + " ".join(extras)
    return label


def normalize_device_from_label(value: str) -> str:
    """
    Ensure we return a pure serial device string.
    """
    if not value:
        return ""
    s = value.strip()
    if s == "(no ports)":
        return ""
    m = re.search(r"\((COM\d+)\)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    if s.upper().startswith("COM"):
        return s.split()[0]
    if s.startswith("/dev/"):
        return s.split()[0]
    if " – " in s:
        return s.split(" – ", 1)[0].strip()
    return s


# ------------- Settings persistence -------------
DEFAULT_SETTINGS = {
    "conn": {
        "mode": "TCP",                 # "TCP" or "RTU"
        "unit_id": "1",
        "timeout": str(DEFAULT_TIMEOUT),
        "tcp": {"host": "127.0.0.1", "port": "502"},
        "rtu": {"port": "COM3", "baud": "9600", "bytesize": "8", "parity": "N", "stopbits": "1"},
    },
    "read": {
        "fc": "Holding Registers",
        "addr": "0",
        "count": "10",
        "decode_type": "raw16",
        "byte_order": "Big",
        "word_order": "Big"
    },
    "write": {
        "target": "Holding Registers",
        "addr": "0",
        "values": "0",
        "encode_type": "raw16",
        "byte_order": "Big",
        "word_order": "Big"
    },
    "poll": {
        "enabled": False,
        "interval": "1.0"
    },
    "log": {
        "filter": "All",
        "max_lines": "2000"
    },
    "display": {
        "num_base": "Dec",      # "Dec", "Hex", "Both"
        "bin_group": "16"       # "16", "8", "4"
    },
    "window": {
        "geometry": "1230x720"
    }
}


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            # merge defaults to ensure new keys exist
            def merge(dflt, data_):
                if isinstance(dflt, dict):
                    out = {}
                    for k, v in dflt.items():
                        out[k] = merge(v, data_.get(k, v)) if isinstance(v, dict) else data_.get(k, v)
                    return out
                return data_ if data_ is not None else dflt
            return merge(DEFAULT_SETTINGS, data)
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass


# ------------- Formatting & parsing helpers for Dec/Hex/Bin -------------
def parse_int_token(token: str, base_mode: str) -> int:
    """
    Parse integer token according to base_mode:
    - "Hex": interpret as hex (allow optional 0x, support -0x..)
    - "Dec": interpret as decimal
    - "Both": default decimal unless explicit 0x prefix
    """
    t = token.strip()
    if not t:
        raise ValueError("Empty token")
    neg = False
    if t.startswith(("+", "-")):
        neg = t[0] == "-"
        t = t[1:].strip()
    if t.lower().startswith("0x"):
        val = int(t, 16)
    else:
        if base_mode == "Hex":
            val = int(t, 16)
        else:
            val = int(t, 10)
    return -val if neg else val


def format_addr(addr: int, base_mode: str) -> str:
    if base_mode == "Hex":
        return f"0x{addr:04X}"
    if base_mode == "Both":
        return f"{addr} (0x{addr:04X})"
    return str(addr)


def is_int_dtype(dtype: str) -> bool:
    return dtype in ("raw16", "int16", "uint16", "int32", "uint32", "int64", "uint64")


def dtype_bit_width(dtype: str) -> int:
    return {
        "raw16": 16,
        "int16": 16, "uint16": 16,
        "int32": 32, "uint32": 32, "float32": 32,
        "int64": 64, "uint64": 64, "float64": 64,
    }[dtype]


def dtype_step_words(dtype: str) -> int:
    return {
        "raw16": 1,
        "int16": 1, "uint16": 1,
        "int32": 2, "uint32": 2, "float32": 2,
        "int64": 4, "uint64": 4, "float64": 4,
    }[dtype]


def format_value_for_display(val, dtype: str, base_mode: str) -> str:
    """Format decoded value for the Value column."""
    if dtype in ("float32", "float64"):
        return str(val)  # floats always decimal for readability
    if dtype == "raw16":
        v = int(val) & 0xFFFF
        if base_mode == "Hex":
            return f"0x{v:04X}"
        if base_mode == "Both":
            return f"{v} (0x{v:04X})"
        return str(v)
    if is_int_dtype(dtype):
        v = int(val)
        bits = dtype_bit_width(dtype)
        if base_mode == "Hex":
            mask = (1 << bits) - 1
            v_hex = v & mask if not (dtype.startswith("int") and v < 0) else (v + (1 << bits)) & mask
            return f"0x{v_hex:X}"
        if base_mode == "Both":
            mask = (1 << bits) - 1
            v_hex = v & mask if not (dtype.startswith("int") and v < 0) else (v + (1 << bits)) & mask
            return f"{v} (0x{v_hex:X})"
        return str(v)
    return str(val)


def format_bool_value(b: bool, base_mode: str) -> str:
    if base_mode == "Hex":
        return "0x1" if b else "0x0"
    if base_mode == "Both":
        return "1 (0x1)" if b else "0 (0x0)"
    return "1" if b else "0"


def group_bin(s: str, mode: str) -> str:
    """Group a 16-bit binary string by mode: '16' (contiguous), '8', or '4'."""
    s = s.zfill(16)
    if mode == "8":
        return f"{s[:8]} {s[8:]}"
    if mode == "4":
        return f"{s[:4]} {s[4:8]} {s[8:12]} {s[12:]}"
    return s


def format_chunk_hex(chunk_regs: list[int]) -> str:
    return " ".join(f"0x{r:04X}" for r in chunk_regs)


def format_chunk_bin(chunk_regs: list[int], bin_group: str) -> str:
    return " ".join(group_bin(f"{r:016b}", bin_group) for r in chunk_regs)


def build_regs_from_bitpattern(int_val: int, total_bits: int, byte_order: str, word_order: str) -> list[int]:
    """
    Construct 16-bit registers from a raw integer bit pattern (for float32/64 hex entry).
    - total_bits: 32 or 64
    - byte_order: 'big' or 'little' (within each 16-bit word)
    - word_order: 'big' or 'little' (order of 16-bit words)
    """
    mask = (1 << total_bits) - 1
    val = int_val & mask
    nbytes = total_bits // 8
    be = val.to_bytes(nbytes, byteorder="big", signed=False)  # bytes in big-endian bit significance
    # split to 16-bit words
    words = [list(be[i:i+2]) for i in range(0, len(be), 2)]
    # byte order within word
    if byte_order.lower().startswith("l"):
        for w in words:
            if len(w) == 2:
                w[0], w[1] = w[1], w[0]
    # word order across words
    if word_order.lower().startswith("l"):
        words = list(reversed(words))
    regs = []
    for w in words:
        b0, b1 = (w + [0, 0])[:2]
        regs.append((b0 << 8) | b1)
    return regs


# ------------- Modbus app -------------
class ModbusApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = load_settings()

        # Window basics
        self.root.title(APP_TITLE)
        self.root.iconbitmap("app.ico")
        geom = self.settings.get("window", {}).get("geometry") or "1230x720"
        try:
            self.root.geometry(geom)
        except Exception:
            self.root.geometry("1230x720")
        self.root.minsize(1230, 720)

        # connection & client
        self.client = None
        self._client_lock = threading.Lock()     # prevent concurrent client I/O
        # Busy & spinner states
        self._io_busy_user = False               # user-initiated I/O (read/write) only
        self._spinner_user_active = False
        self._spinner_user_msg = "Working…"
        self._spinner_poll_active = False        # non-blocking spinner after polling errors
        self._spinner_poll_msg = "Polling… (no response)"
        self.conn_mode = tk.StringVar(value=self.settings["conn"]["mode"])  # TCP / RTU
        self.status_var = tk.StringVar(value="Disconnected")

        # Common vars
        self.unit_id_var = tk.StringVar(value=self.settings["conn"]["unit_id"])
        self.timeout_var = tk.StringVar(value=self.settings["conn"]["timeout"])

        # TCP vars
        self.tcp_host_var = tk.StringVar(value=self.settings["conn"]["tcp"]["host"])
        self.tcp_port_var = tk.StringVar(value=self.settings["conn"]["tcp"]["port"])

        # RTU vars
        rtu_port_saved = normalize_device_from_label(self.settings["conn"]["rtu"]["port"])
        self.rtu_port_var = tk.StringVar(value=rtu_port_saved)
        self.rtu_baud_var = tk.StringVar(value=self.settings["conn"]["rtu"]["baud"])
        self.rtu_bytesize_var = tk.StringVar(value=self.settings["conn"]["rtu"]["bytesize"])
        self.rtu_parity_var = tk.StringVar(value=self.settings["conn"]["rtu"]["parity"])
        self.rtu_stopbits_var = tk.StringVar(value=self.settings["conn"]["rtu"]["stopbits"])

        # Read controls
        self.read_fc_var = tk.StringVar(value=self.settings["read"]["fc"])
        self.read_addr_var = tk.StringVar(value=self.settings["read"]["addr"])
        self.read_count_var = tk.StringVar(value=self.settings["read"]["count"])

        self.decode_type_var = tk.StringVar(value=self.settings["read"]["decode_type"])
        self.byte_order_var = tk.StringVar(value=self.settings["read"]["byte_order"])
        self.word_order_var = tk.StringVar(value=self.settings["read"]["word_order"])

        # Write controls
        self.write_target_var = tk.StringVar(value=self.settings["write"]["target"])
        self.write_addr_var = tk.StringVar(value=self.settings["write"]["addr"])
        self.write_values_var = tk.StringVar(value=self.settings["write"]["values"])
        self.encode_type_var = tk.StringVar(value=self.settings["write"]["encode_type"])
        self.w_byte_order_var = tk.StringVar(value=self.settings["write"]["byte_order"])
        self.w_word_order_var = tk.StringVar(value=self.settings["write"]["word_order"])

        # Polling
        self.polling = False
        self.poll_after_id = None
        self.poll_interval_var = tk.StringVar(value=self.settings["poll"]["interval"])

        # Logging
        self._log_buffer: list[tuple[str, str, str]] = []  # (ts, level, message)
        self.log_filter_var = tk.StringVar(value=self.settings.get("log", {}).get("filter", "All"))
        self.log_max_lines_var = tk.StringVar(value=self.settings.get("log", {}).get("max_lines", "2000"))

        # Display base (Dec/Hex/Both) and binary grouping
        self.num_base_var = tk.StringVar(value=self.settings.get("display", {}).get("num_base", "Dec"))
        self.bin_group_var = tk.StringVar(value=self.settings.get("display", {}).get("bin_group", "16"))

        # Results model store for dynamic re-rendering
        self._results_model = []  # list of dicts: {idx, addr, dtype, value, chunk}

        # UI refs set later
        self.port_combo = None
        self.conn_dot = None
        self.conn_dot_item = None
        self.btn_poll = None
        self.btn_read = None
        self.btn_write = None
        self.spinner = None
        self.spinner_label = None
        self.log_text = None
        self.entry_poll = None
        self.filter_combo = None
        self.max_lines_entry = None

        # Mapping from display label -> device path for serial ports
        self._port_label_to_device = {}

        # Build UI
        self._build_ui()

        # Key bindings
        self.root.bind("<Escape>", lambda e: self._on_escape())
        self.root.bind("<Return>", lambda e: self._on_enter())

        # initialize connection dot
        self._update_conn_dot(False)

        # Save settings on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Prepare serial ports list at startup
        self.refresh_serial_ports()

        # Poll button label from last state
        self._set_poll_button_state(started=False)

        # Initial log banner
        self.log("Application started.", "INFO")

    # ----------- UI ----------
    def _build_ui(self):
        # Connection frame
        conn = ttk.LabelFrame(self.root, text="Connection")
        conn.pack(fill="x", padx=10, pady=8)

        # Mode
        ttk.Radiobutton(conn, text="Modbus TCP", variable=self.conn_mode, value="TCP", command=self._on_mode_change)\
            .grid(row=0, column=0, padx=(10, 6), pady=6, sticky="w")
        ttk.Radiobutton(conn, text="Modbus RTU", variable=self.conn_mode, value="RTU", command=self._on_mode_change)\
            .grid(row=0, column=1, padx=(0, 10), pady=6, sticky="w")

        # TCP fields
        ttk.Label(conn, text="Host").grid(row=1, column=0, sticky="e", padx=10)
        ttk.Entry(conn, textvariable=self.tcp_host_var, width=18).grid(row=1, column=1, sticky="w")
        ttk.Label(conn, text="Port").grid(row=1, column=2, sticky="e", padx=10)
        ttk.Entry(conn, textvariable=self.tcp_port_var, width=8).grid(row=1, column=3, sticky="w")

        # RTU fields
        ttk.Label(conn, text="Serial Port").grid(row=2, column=0, sticky="e", padx=10)
        self.port_combo = ttk.Combobox(
            conn, textvariable=self.rtu_port_var, values=[], width=28, state="readonly"
        )
        self.port_combo.grid(row=2, column=1, columnspan=2, sticky="w")
        self.port_combo.bind("<<ComboboxSelected>>", self._on_port_selected)

        btn_refresh_ports = ttk.Button(conn, text="Refresh", command=self.refresh_serial_ports)
        btn_refresh_ports.grid(row=2, column=3, sticky="w", padx=(6, 0))

        ttk.Label(conn, text="Baud").grid(row=2, column=4, sticky="e", padx=10)
        self.baud_combo = ttk.Combobox(
            conn, textvariable=self.rtu_baud_var, values=BAUD_RATES, width=10, state="readonly"
        )
        self.baud_combo.grid(row=2, column=5, sticky="w")

        ttk.Label(conn, text="Data bits").grid(row=2, column=6, sticky="e", padx=10)
        ttk.Entry(conn, textvariable=self.rtu_bytesize_var, width=6).grid(row=2, column=7, sticky="w")
        ttk.Label(conn, text="Parity").grid(row=2, column=8, sticky="e", padx=10)
        ttk.Combobox(conn, textvariable=self.rtu_parity_var, values=["N", "E", "O"], width=4, state="readonly")\
            .grid(row=2, column=9, sticky="w")
        ttk.Label(conn, text="Stop bits").grid(row=2, column=10, sticky="e", padx=10)
        ttk.Entry(conn, textvariable=self.rtu_stopbits_var, width=6).grid(row=2, column=11, sticky="w")

        # Common: unit + timeout
        ttk.Label(conn, text="Unit ID").grid(row=1, column=4, sticky="e", padx=10)
        ttk.Entry(conn, textvariable=self.unit_id_var, width=6).grid(row=1, column=5, sticky="w")
        ttk.Label(conn, text="Timeout (s)").grid(row=1, column=6, sticky="e", padx=10)
        ttk.Entry(conn, textvariable=self.timeout_var, width=8).grid(row=1, column=7, sticky="w")

        ttk.Button(conn, text="Connect", command=self.connect).grid(row=1, column=8, padx=10, sticky="e")
        ttk.Button(conn, text="Disconnect", command=self.disconnect).grid(row=1, column=9, padx=2, sticky="w")

        ttk.Label(conn, textvariable=self.status_var, foreground="#006400").grid(row=0, column=8, columnspan=2, sticky="e", padx=10)

        # Dataset (read/write)
        dataset = ttk.LabelFrame(self.root, text="Read/Write")
        dataset.pack(fill="x", padx=10, pady=(2, 0))

        # Base selector (Dec / Hex / Both) + Binary grouping
        base_frame = ttk.Frame(dataset)
        base_frame.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(base_frame, text="Number base:").pack(side="left")
        ttk.Radiobutton(base_frame, text="Dec", variable=self.num_base_var, value="Dec",
                        command=self._on_display_change).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(base_frame, text="Hex", variable=self.num_base_var, value="Hex",
                        command=self._on_display_change).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(base_frame, text="Both", variable=self.num_base_var, value="Both",
                        command=self._on_display_change).pack(side="left", padx=(6, 12))
        ttk.Label(base_frame, text="Binary grouping:").pack(side="left")
        self.bin_group_combo = ttk.Combobox(base_frame, textvariable=self.bin_group_var,
                                            values=["16", "8", "4"], state="readonly", width=4)
        self.bin_group_combo.pack(side="left", padx=(6, 0))
        self.bin_group_combo.bind("<<ComboboxSelected>>", lambda e: self._on_display_change())

        # --- Read panel ---
        readf = ttk.Frame(dataset)
        readf.pack(fill="x", padx=6, pady=6)

        ttk.Label(readf, text="Read from").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Combobox(readf, textvariable=self.read_fc_var, values=[
            "Coils", "Discrete Inputs", "Holding Registers", "Input Registers"
        ], state="readonly", width=20).grid(row=0, column=1, sticky="w")

        ttk.Label(readf, text="Address").grid(row=0, column=2, sticky="e", padx=10)
        ttk.Entry(readf, textvariable=self.read_addr_var, width=12).grid(row=0, column=3, sticky="w")

        ttk.Label(readf, text="Count").grid(row=0, column=4, sticky="e", padx=10)
        ttk.Entry(readf, textvariable=self.read_count_var, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(readf, text="Decode as").grid(row=0, column=6, sticky="e", padx=10)
        ttk.Combobox(readf, textvariable=self.decode_type_var, values=[
            "raw16", "int16", "uint16", "int32", "uint32", "float32", "int64", "uint64", "float64"
        ], state="readonly", width=10).grid(row=0, column=7, sticky="w")

        ttk.Label(readf, text="Byte order").grid(row=0, column=8, sticky="e", padx=10)
        ttk.Combobox(readf, textvariable=self.byte_order_var, values=["Big", "Little"], state="readonly", width=8)\
            .grid(row=0, column=9, sticky="w")
        ttk.Label(readf, text="Word order").grid(row=0, column=10, sticky="e", padx=10)
        ttk.Combobox(readf, textvariable=self.word_order_var, values=["Big", "Little"], state="readonly", width=8)\
            .grid(row=0, column=11, sticky="w")

        self.btn_read = ttk.Button(readf, text="Read", command=self.on_read)
        self.btn_read.grid(row=0, column=12, padx=10)

        # --- Write panel ---
        writef = ttk.Frame(dataset)
        writef.pack(fill="x", padx=6, pady=(6, 6))

        ttk.Label(writef, text="Write to").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Combobox(writef, textvariable=self.write_target_var, values=[
            "Coils", "Holding Registers"
        ], state="readonly", width=20).grid(row=0, column=1, sticky="w")

        ttk.Label(writef, text="Address").grid(row=0, column=2, sticky="e", padx=10)
        ttk.Entry(writef, textvariable=self.write_addr_var, width=12).grid(row=0, column=3, sticky="w")

        ttk.Label(writef, text="Value(s)").grid(row=0, column=4, sticky="e", padx=10)
        ttk.Entry(writef, textvariable=self.write_values_var, width=40).grid(row=0, column=5, sticky="w")

        ttk.Label(writef, text="Encode as").grid(row=0, column=6, sticky="e", padx=10)
        ttk.Combobox(writef, textvariable=self.encode_type_var, values=[
            "raw16", "int16", "uint16", "int32", "uint32", "float32", "int64", "uint64", "float64"
        ], state="readonly", width=10).grid(row=0, column=7, sticky="w")

        ttk.Label(writef, text="Byte order").grid(row=0, column=8, sticky="e", padx=10)
        ttk.Combobox(writef, textvariable=self.w_byte_order_var, values=["Big", "Little"], state="readonly", width=8)\
            .grid(row=0, column=9, sticky="w")
        ttk.Label(writef, text="Word order").grid(row=0, column=10, sticky="e", padx=10)
        ttk.Combobox(writef, textvariable=self.w_word_order_var, values=["Big", "Little"], state="readonly", width=8)\
            .grid(row=0, column=11, sticky="w")

        self.btn_write = ttk.Button(writef, text="Write", command=self.on_write)
        self.btn_write.grid(row=0, column=12, padx=10)

        # Results table
        res_frame = ttk.LabelFrame(self.root, text="Results")
        res_frame.pack(fill="both", expand=True, padx=10, pady=8)

        self.tree = ttk.Treeview(res_frame, columns=("idx", "addr", "value", "hex", "bin", "raw"), show="headings", height=16)
        self.tree.heading("idx", text="#")
        self.tree.heading("addr", text="Address")
        self.tree.heading("value", text="Value")
        self.tree.heading("hex", text="Hex (raw)")
        self.tree.heading("bin", text="Bin (raw)")
        self.tree.heading("raw", text="Raw")
        self.tree.column("idx", width=50, anchor="center")
        self.tree.column("addr", width=140, anchor="center")
        self.tree.column("value", width=220, anchor="w")
        self.tree.column("hex", width=220, anchor="w")
        self.tree.column("bin", width=300, anchor="w")
        self.tree.column("raw", width=200, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(res_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")

        # Bottom bar with CSV button, polling controls, spinner, and connection dot
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(bottom, text="Save table as CSV…", command=self.save_csv).pack(side="left")

        # Polling controls
        poll_frame = ttk.Frame(bottom)
        poll_frame.pack(side="left", padx=(20, 0))
        ttk.Label(poll_frame, text="Poll (s):").pack(side="left", padx=(0, 6))
        self.entry_poll = ttk.Entry(poll_frame, textvariable=self.poll_interval_var, width=8)
        self.entry_poll.pack(side="left")
        self.btn_poll = ttk.Button(poll_frame, text="Start Polling", command=self.toggle_polling)
        self.btn_poll.pack(side="left", padx=(8, 0))

        # Spinner (indeterminate progress) – initially hidden
        self.spinner_label = ttk.Label(bottom, text="Working…")
        self.spinner = ttk.Progressbar(bottom, orient="horizontal", mode="indeterminate", length=140)

        # Right-side connection indicator (small colored dot)
        self.conn_dot = tk.Canvas(bottom, width=16, height=16, highlightthickness=0, bg=self.root.cget("bg"))
        self.conn_dot.pack(side="right")
        self.conn_dot_item = self.conn_dot.create_oval(2, 2, 14, 14, fill="#c00000", outline="#a00000")  # red

        # ----- LOG PANEL -----
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))

        # Toolbar above log
        toolbar = ttk.Frame(log_frame)
        toolbar.pack(fill="x", padx=6, pady=4)

        ttk.Label(toolbar, text="Filter:").pack(side="left")
        self.filter_combo = ttk.Combobox(toolbar, textvariable=self.log_filter_var,
                                         values=LOG_FILTER_VALUES, state="readonly", width=8)
        self.filter_combo.pack(side="left", padx=(4, 10))
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_log_text())

        ttk.Label(toolbar, text="Max lines:").pack(side="left")
        self.max_lines_entry = ttk.Entry(toolbar, textvariable=self.log_max_lines_var, width=8)
        self.max_lines_entry.pack(side="left", padx=(4, 10))

        ttk.Button(toolbar, text="Clear", command=self.clear_log).pack(side="right", padx=(0, 12))
        ttk.Button(toolbar, text="Save…", command=self.export_log).pack(side="right", padx=(0, 6))
        ttk.Button(toolbar, text="Copy", command=self.copy_log).pack(side="right", padx=(0, 6))

        # Log text widget
        self.log_text = tk.Text(
            log_frame,
            height=10,
            wrap="word",
            state="disabled",
            background="#1e1e1e",
            foreground="#cccccc",
            font=("Consolas", 10)
        )
        self.log_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")
        self.log_text.configure(yscrollcommand=scroll.set)

        # Configure color tags
        self.log_text.tag_configure("TS", foreground="#888888")
        self.log_text.tag_configure("INFO", foreground="#d0d0d0")
        self.log_text.tag_configure("WARN", foreground="#ffd166")
        self.log_text.tag_configure("ERROR", foreground="#ff6b6b")

        self._on_mode_change()

    # ---------- Serial port helpers ----------
    def _get_serial_ports(self) -> list:
        if list_ports is None:
            return []
        try:
            return list(list_ports.comports())
        except Exception:
            return []

    def refresh_serial_ports(self):
        ports = self._get_serial_ports()
        labels = []
        self._port_label_to_device.clear()
        for p in ports:
            lbl = _format_port_entry(p)
            labels.append(lbl)
            self._port_label_to_device[lbl] = p.device
        if not labels:
            labels = ["(no ports)"]
        self.port_combo["values"] = labels

        current_dev = normalize_device_from_label(self.rtu_port_var.get() or "")
        if current_dev:
            for lbl, dev in self._port_label_to_device.items():
                if dev == current_dev:
                    self.port_combo.set(lbl)
                    break
            else:
                self.port_combo.set(labels[0])
                if labels[0] != "(no ports)":
                    self.rtu_port_var.set(self._port_label_to_device[labels[0]])
                else:
                    self.rtu_port_var.set("")
        else:
            self.port_combo.set(labels[0])
            if labels[0] != "(no ports)":
                self.rtu_port_var.set(self._port_label_to_device[labels[0]])
            else:
                self.rtu_port_var.set("")
        self._save_settings_lazy()

    def _on_port_selected(self, event=None):
        lbl = self.port_combo.get().strip()
        dev = self._port_label_to_device.get(lbl, "")
        if lbl == "(no ports)":
            dev = ""
        self.rtu_port_var.set(dev)
        if dev:
            self.conn_mode.set("RTU")
            self._on_mode_change()
        self._save_settings_lazy()

    # ---------- Events ----------
    def _on_mode_change(self):
        mode = self.conn_mode.get()
        if mode == "TCP":
            self.status_var.set("TCP mode selected (host/port)")
            self.log("Switched to TCP mode.", "INFO")
        else:
            self.status_var.set("RTU mode selected (serial)")
            self.log("Switched to RTU mode.", "INFO")
            self.refresh_serial_ports()
        self._save_settings_lazy()

    def _on_display_change(self):
        self._render_results()  # re-render table with new base / bin grouping
        self._save_settings_lazy()

    def _on_escape(self):
        self.disconnect()
        self.status_var.set("Aborted / Disconnected")

    def _on_enter(self):
        self.on_read()

    # ---------- Connection ----------
    def connect(self):
        self.disconnect()
        try:
            unit = safe_int(self.unit_id_var.get(), 1)
            timeout = float(self.timeout_var.get() or DEFAULT_TIMEOUT)

            if self.conn_mode.get() == "TCP":
                host = self.tcp_host_var.get().strip()
                port = safe_int(self.tcp_port_var.get(), 502)
                with self._client_lock:
                    self.client = ModbusTcpClient(host=host, port=port, timeout=timeout)
                    ok = self.client.connect()
                if not ok:
                    raise RuntimeError(f"Failed to connect TCP {host}:{port}")
                self.status_var.set(f"Connected TCP {host}:{port} | Unit {unit}")
                self.log(f"Connected successfully: {host}:{port} (unit {unit})", "INFO")

            else:
                port = normalize_device_from_label(self.rtu_port_var.get().strip())
                if not port:
                    raise RuntimeError("No serial port selected. Click Refresh and pick a port.")
                baud = safe_int(self.rtu_baud_var.get(), 9600)
                bytesize = safe_int(self.rtu_bytesize_var.get(), 8)
                parity = self.rtu_parity_var.get().strip().upper()[:1] or "N"
                stopbits = safe_int(self.rtu_stopbits_var.get(), 1)
                with self._client_lock:
                    self.client = ModbusSerialClient(
                        port=port,
                        baudrate=baud,
                        bytesize=bytesize,
                        parity=parity,
                        stopbits=stopbits,
                        timeout=timeout
                    )
                    ok = self.client.connect()
                if not ok:
                    raise RuntimeError(f"Failed to open serial {port}")
                self.status_var.set(f"Connected RTU {port}@{baud} | Unit {unit}")
                self.log(f"Connected successfully: {port}@{baud} (unit {unit})", "INFO")

            self._update_conn_dot(True)
            self._save_settings()

        except Exception as e:
            with self._client_lock:
                self.client = None
            self._update_conn_dot(False)
            self.log(f"Connect error: {e}", "ERROR")
            show_error("Connect error", e)

    def disconnect(self):
        if self.polling:
            self._stop_polling()
        try:
            with self._client_lock:
                if self.client:
                    self.client.close()
        except Exception:
            pass
        with self._client_lock:
            self.client = None
        self.status_var.set("Disconnected")
        self._update_conn_dot(False)
        self.log("Disconnected from device.", "INFO")
        self._save_settings_lazy()

    # ---------- Threading & UI helpers ----------
    def run_in_thread(self, func):
        t = threading.Thread(target=func, daemon=True)
        t.start()

    def _update_controls_enabled(self):
        try:
            state = "disabled" if self._io_busy_user else "normal"
            if self.btn_read:
                self.btn_read.config(state=state)
            if self.btn_write:
                self.btn_write.config(state=state)
            if self.entry_poll:
                self.entry_poll.config(state=state)
            if self.btn_poll:
                self.btn_poll.config(state=state)
        except Exception:
            pass

    def _refresh_spinner(self):
        try:
            active = False
            message = "Working…"
            if self._spinner_user_active:
                active = True
                message = self._spinner_user_msg
            elif self._spinner_poll_active:
                active = True
                message = self._spinner_poll_msg

            if active:
                self.spinner_label.config(text=message)
                self.spinner.pack_forget()
                self.spinner_label.pack_forget()
                self.spinner_label.pack(side="right", padx=(8, 6))
                self.spinner.pack(side="right")
                self.spinner.start(10)
            else:
                self.spinner.stop()
                self.spinner.pack_forget()
                self.spinner_label.pack_forget()
        except Exception:
            pass

    def _spinner_user(self, on: bool, message: str = "Working…"):
        self._io_busy_user = on
        self._spinner_user_active = on
        self._spinner_user_msg = message or "Working…"
        self._update_controls_enabled()
        self._refresh_spinner()

    def _spinner_poll_error(self, on: bool, message: str = "Polling… (no response)"):
        self._spinner_poll_active = on
        self._spinner_poll_msg = message or "Polling… (no response)"
        self._refresh_spinner()

    def _set_results(self, model_rows):
        """Store model rows (list of dicts) and render based on current display options."""
        self._results_model = model_rows or []
        self._render_results()

    def _render_results(self):
        """Render self._results_model to the tree, respecting num_base and bin_group."""
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        base_mode = self.num_base_var.get()
        bin_group = self.bin_group_var.get()
        for row in self._results_model:
            idx = row["idx"]
            a = format_addr(row["addr"], base_mode)
            val_str = format_value_for_display(row["value"], row["dtype"], base_mode)
            hexs = format_chunk_hex(row["chunk"])
            bins = format_chunk_bin(row["chunk"], bin_group)
            raw_str = "[" + ", ".join(str(x) for x in row["chunk"]) + "]"
            self.tree.insert("", "end", values=(idx, a, val_str, hexs, bins, raw_str))

    # ---------- Logging ----------
    def _passes_filter(self, level: str) -> bool:
        sel = (self.log_filter_var.get() or "All").upper()
        if sel == "ALL":
            return True
        level = level.upper()
        try:
            return LEVEL_ORDER[level] >= LEVEL_ORDER[sel]
        except KeyError:
            return True

    def _append_to_log_widget(self, ts: str, level: str, message: str):
        if not self._passes_filter(level):
            return
        self.log_text.configure(state="normal")
        start_index = self.log_text.index("end-1c")
        line = f"[{ts}] [{level}] {message}\n"
        self.log_text.insert("end", line)
        end_index = self.log_text.index("end-1c")
        self.log_text.tag_add(level, start_index, end_index)
        try:
            ts_end = self.log_text.search("]", start_index, stopindex=end_index)
            if ts_end:
                self.log_text.tag_add("TS", start_index, f"{ts_end}+1c")
        except Exception:
            pass
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _trim_log_buffer(self):
        max_lines = safe_int(self.log_max_lines_var.get(), 2000)
        trimmed = False
        if max_lines <= 0:
            max_lines = 2000
        while len(self._log_buffer) > max_lines:
            self._log_buffer.pop(0)
            trimmed = True
        if trimmed:
            self._rebuild_log_text()

    def _rebuild_log_text(self):
        if not self.log_text:
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for ts, lvl, msg in self._log_buffer:
            if self._passes_filter(lvl):
                start_index = self.log_text.index("end-1c")
                line = f"[{ts}] [{lvl}] {msg}\n"
                self.log_text.insert("end", line)
                end_index = self.log_text.index("end-1c")
                self.log_text.tag_add(lvl, start_index, end_index)
                try:
                    ts_end = self.log_text.search("]", start_index, stopindex=end_index)
                    if ts_end:
                        self.log_text.tag_add("TS", start_index, f"{ts_end}+1c")
                except Exception:
                    pass
        self.log_text.configure(state="disabled")
        self.log_text.see("end")
        self._save_settings_lazy()

    def log(self, message: str, level="INFO"):
        level = (level or "INFO").upper()
        if level not in LEVEL_ORDER:
            level = "INFO"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_buffer.append((ts, level, message))
        self._trim_log_buffer()
        try:
            self.root.after(0, lambda: self._append_to_log_widget(ts, level, message))
        except RuntimeError:
            pass

    def clear_log(self):
        self._log_buffer.clear()
        if self.log_text:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")
        self.log("Log cleared by user.", "INFO")

    def copy_log(self):
        try:
            text = self.log_text.get("sel.first", "sel.last")
            if not text.strip():
                text = self.log_text.get("1.0", "end").strip("\n")
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.log("Log copied to clipboard.", "INFO")
        except Exception:
            pass

    def export_log(self):
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                title="Export log to file"
            )
            if not path:
                return
            filtered_lines = []
            for ts, lvl, msg in self._log_buffer:
                if self._passes_filter(lvl):
                    filtered_lines.append(f"[{ts}] [{lvl}] {msg}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(filtered_lines) + ("\n" if filtered_lines else ""))
            self.log(f"Log exported to {path}", "INFO")
            messagebox.showinfo("Export log", f"Saved: {path}")
        except Exception as e:
            self.log(f"Export log error: {e}", "ERROR")
            show_error("Export log error", e)

    # ---------- Read ----------
    def on_read(self):
        if not ensure_connected(self.client):
            show_error("Not connected", "Please connect first.")
            self.log("Read requested but not connected.", "WARN")
            return

        if self._io_busy_user:
            messagebox.showinfo("Busy", "Another operation is in progress. Please wait.")
            self.log("Read ignored: another user operation is in progress.", "WARN")
            return

        # Capture UI state in main thread
        base_mode = self.num_base_var.get()
        try:
            addr = parse_int_token(self.read_addr_var.get(), base_mode)
        except Exception as e:
            show_error("Read error", f"Invalid address: {e}")
            return

        unit = safe_int(self.unit_id_var.get(), 1)
        count = safe_int(self.read_count_var.get(), 1)
        fc = self.read_fc_var.get()
        dtype = self.decode_type_var.get()
        byte_order = self.byte_order_var.get().lower()
        word_order = self.word_order_var.get().lower()

        def task():
            self.root.after(0, lambda: self._spinner_user(True, "Reading…"))
            try:
                model_rows = []

                if fc == "Coils":
                    with self._client_lock:
                        rr = self.client.read_coils(address=addr, count=count, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)
                    bits = list(rr.bits or [])[:count]
                    for i, b in enumerate(bits):
                        model_rows.append({"idx": i, "addr": addr + i, "dtype": "raw1", "value": 1 if b else 0, "chunk": [1 if b else 0]})

                elif fc == "Discrete Inputs":
                    with self._client_lock:
                        rr = self.client.read_discrete_inputs(address=addr, count=count, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)
                    bits = list(rr.bits or [])[:count]
                    for i, b in enumerate(bits):
                        model_rows.append({"idx": i, "addr": addr + i, "dtype": "raw1", "value": 1 if b else 0, "chunk": [1 if b else 0]})

                elif fc in ("Holding Registers", "Input Registers"):
                    if fc == "Holding Registers":
                        with self._client_lock:
                            rr = self.client.read_holding_registers(address=addr, count=count, device_id=unit)
                    else:
                        with self._client_lock:
                            rr = self.client.read_input_registers(address=addr, count=count, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)
                    regs = list(rr.registers or [])
                    # decode in chunks according to dtype
                    step = dtype_step_words(dtype)
                    values_idx = 0
                    for i in range(0, len(regs) - (len(regs) % step), step):
                        chunk = regs[i:i+step]
                        try:
                            with self._client_lock:
                                val = self.client.convert_from_registers(
                                    chunk,
                                    data_type=dtype,
                                    word_order=word_order,
                                    byte_order=byte_order
                                )
                        except Exception as e:
                            raise RuntimeError(f"Decode failed ({dtype}) at index {i//step}: {e}") from e
                        model_rows.append({"idx": values_idx, "addr": addr + i, "dtype": dtype, "value": val, "chunk": chunk})
                        values_idx += 1
                else:
                    raise ValueError("Unsupported function")

                self.root.after(0, lambda: self._set_results(model_rows))
                self.root.after(0, lambda: self.log(f"Read OK: {fc} @ {addr}, count={count}", "INFO"))
                self.root.after(0, self._save_settings_lazy)

            except Exception as e:
                self.root.after(0, lambda: self.log(f"Read error: {e}", "ERROR"))
                self.root.after(0, lambda: show_error("Read error", e))
            finally:
                self.root.after(0, lambda: self._spinner_user(False))

        self.run_in_thread(task)

    # ---------- Write ----------
    def on_write(self):
        if not ensure_connected(self.client):
            show_error("Not connected", "Please connect first.")
            self.log("Write requested but not connected.", "WARN")
            return

        if self._io_busy_user:
            messagebox.showinfo("Busy", "Another operation is in progress. Please wait.")
            self.log("Write ignored: another user operation is in progress.", "WARN")
            return

        # Capture UI state in main thread
        base_mode = self.num_base_var.get()
        try:
            addr = parse_int_token(self.write_addr_var.get(), base_mode)
        except Exception as e:
            show_error("Write error", f"Invalid address: {e}")
            return

        unit = safe_int(self.unit_id_var.get(), 1)
        target = self.write_target_var.get()
        values_str = self.write_values_var.get().strip()
        dtype = self.encode_type_var.get()
        byte_order = self.w_byte_order_var.get().lower()
        word_order = self.w_word_order_var.get().lower()

        if not values_str:
            show_error("Write error", "Provide Value(s). For multiple, separate by commas.")
            self.log("Write aborted: no values provided.", "WARN")
            return

        def task():
            self.root.after(0, lambda: self._spinner_user(True, "Writing…"))
            try:
                if target == "Coils":
                    vals = [str2bool(x) for x in values_str.split(",")]
                    with self._client_lock:
                        if len(vals) == 1:
                            rr = self.client.write_coil(address=addr, value=vals[0], device_id=unit)
                        else:
                            rr = self.client.write_coils(address=addr, values=vals, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)

                elif target == "Holding Registers":
                    regs = []
                    parts = [p for p in values_str.split(",") if p.strip() != ""]
                    if dtype == "raw16":
                        for p in parts:
                            v = parse_int_token(p, base_mode) & 0xFFFF
                            regs.append(v)
                    elif dtype in ("int16", "uint16", "int32", "uint32", "int64", "uint64"):
                        for p in parts:
                            v = parse_int_token(p, base_mode)
                            with self._client_lock:
                                chunk = self.client.convert_to_registers(
                                    v,
                                    data_type=dtype,
                                    word_order=word_order,
                                    byte_order=byte_order
                                )
                            regs.extend(chunk)
                    elif dtype in ("float32", "float64"):
                        bits = 32 if dtype == "float32" else 64
                        for p in parts:
                            p_clean = p.strip()
                            if base_mode == "Hex" or p_clean.lower().startswith("0x"):
                                # Interpret as bit pattern integer
                                v_int = parse_int_token(p_clean, "Hex")
                                chunk = build_regs_from_bitpattern(v_int, bits, byte_order, word_order)
                            else:
                                # Decimal float value
                                val = float(p_clean)
                                with self._client_lock:
                                    chunk = self.client.convert_to_registers(
                                        val,
                                        data_type=dtype,
                                        word_order=word_order,
                                        byte_order=byte_order
                                    )
                            regs.extend(chunk)
                    else:
                        raise ValueError(f"Unsupported encode type: {dtype}")

                    if len(regs) == 0:
                        raise RuntimeError("No registers built from provided values.")

                    with self._client_lock:
                        if len(regs) == 1:
                            rr = self.client.write_register(address=addr, value=regs[0], device_id=unit)
                        else:
                            rr = self.client.write_registers(address=addr, values=regs, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)

                else:
                    raise ValueError("Write target must be Coils or Holding Registers.")

                self.root.after(0, lambda: messagebox.showinfo("Write", "Write successful."))
                self.root.after(0, lambda: self.log(f"Write OK: {target} @ {addr} -> {values_str}", "INFO"))
                self.root.after(0, self._save_settings_lazy)

            except Exception as e:
                self.root.after(0, lambda: self.log(f"Write error: {e}", "ERROR"))
                self.root.after(0, lambda: show_error("Write error", e))
            finally:
                self.root.after(0, lambda: self._spinner_user(False))

        self.run_in_thread(task)

    # ---------- Polling ----------
    def toggle_polling(self):
        if self.polling:
            self._stop_polling()
        else:
            self._start_polling()

    def _start_polling(self):
        if not ensure_connected(self.client):
            show_error("Not connected", "Please connect first, then start polling.")
            self.log("Polling start failed: not connected.", "WARN")
            return
        interval_s = safe_float(self.poll_interval_var.get(), 1.0)
        if interval_s <= 0:
            show_error("Invalid interval", "Polling interval must be > 0.")
            self.log("Polling start failed: invalid interval.", "WARN")
            return
        self.polling = True
        self._set_poll_button_state(started=True)
        self.log(f"Polling started (every {interval_s}s).", "INFO")
        self._spinner_poll_error(False)
        self._poll_once()
        self._save_settings_lazy()

    def _stop_polling(self):
        self.polling = False
        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except Exception:
                pass
            self.poll_after_id = None
        self._set_poll_button_state(started=False)
        self._spinner_poll_error(False)
        self.log("Polling stopped.", "INFO")
        self._save_settings_lazy()

    def _poll_once(self):
        if not self.polling:
            return
        if not ensure_connected(self.client):
            self.log("Polling: not connected.", "WARN")
            self._spinner_poll_error(False)
            interval_s = safe_float(self.poll_interval_var.get(), 1.0)
            self.poll_after_id = self.root.after(max(1, int(interval_s * 1000)), self._poll_once)
            return

        if self._io_busy_user:
            self.poll_after_id = self.root.after(50, self._poll_once)
            return

        base_mode = self.num_base_var.get()
        try:
            addr = parse_int_token(self.read_addr_var.get(), base_mode)
        except Exception:
            addr = safe_int(self.read_addr_var.get(), 0)

        unit = safe_int(self.unit_id_var.get(), 1)
        count = safe_int(self.read_count_var.get(), 1)
        fc = self.read_fc_var.get()
        dtype = self.decode_type_var.get()
        byte_order = self.byte_order_var.get().lower()
        word_order = self.word_order_var.get().lower()
        interval_s = safe_float(self.poll_interval_var.get(), 1.0)
        interval_ms = max(1, int(interval_s * 1000))

        def task():
            try:
                model_rows = []

                if fc == "Coils":
                    with self._client_lock:
                        rr = self.client.read_coils(address=addr, count=count, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)
                    bits = list(rr.bits or [])[:count]
                    for i, b in enumerate(bits):
                        model_rows.append({"idx": i, "addr": addr + i, "dtype": "raw1", "value": 1 if b else 0, "chunk": [1 if b else 0]})

                elif fc == "Discrete Inputs":
                    with self._client_lock:
                        rr = self.client.read_discrete_inputs(address=addr, count=count, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)
                    bits = list(rr.bits or [])[:count]
                    for i, b in enumerate(bits):
                        model_rows.append({"idx": i, "addr": addr + i, "dtype": "raw1", "value": 1 if b else 0, "chunk": [1 if b else 0]})

                elif fc in ("Holding Registers", "Input Registers"):
                    if fc == "Holding Registers":
                        with self._client_lock:
                            rr = self.client.read_holding_registers(address=addr, count=count, device_id=unit)
                    else:
                        with self._client_lock:
                            rr = self.client.read_input_registers(address=addr, count=count, device_id=unit)
                    if rr.isError():
                        raise RuntimeError(rr)
                    regs = list(rr.registers or [])
                    step = dtype_step_words(dtype)
                    values_idx = 0
                    for i in range(0, len(regs) - (len(regs) % step), step):
                        chunk = regs[i:i+step]
                        try:
                            with self._client_lock:
                                val = self.client.convert_from_registers(
                                    chunk,
                                    data_type=dtype,
                                    word_order=word_order,
                                    byte_order=byte_order
                                )
                        except Exception as e:
                            raise RuntimeError(f"Decode failed ({dtype}) at index {i//step}: {e}") from e
                        model_rows.append({"idx": values_idx, "addr": addr + i, "dtype": dtype, "value": val, "chunk": chunk})
                        values_idx += 1
                else:
                    raise ValueError("Unsupported function")

                self.root.after(0, lambda: self._set_results(model_rows))
                self.root.after(0, lambda: self.log(f"Poll OK: {fc} @ {addr}, count={count}", "INFO"))
                self.root.after(0, lambda: self._spinner_poll_error(False))

            except Exception as e:
                self.root.after(0, lambda: self.log(f"Poll error: {e}", "ERROR"))
                self.root.after(0, lambda: self._spinner_poll_error(True, "Polling… (no response)"))
            finally:
                if self.polling:
                    self.poll_after_id = self.root.after(interval_ms, self._poll_once)

        self.run_in_thread(task)

    def _set_poll_button_state(self, started: bool):
        if self.btn_poll:
            self.btn_poll.config(text="Stop Polling" if started else "Start Polling")

    # ---------- Table/CSV ----------
    def save_csv(self):
        try:
            if not self.tree.get_children():
                messagebox.showinfo("Save CSV", "No data to save.")
                self.log("Save CSV skipped: no data.", "WARN")
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save results as CSV"
            )
            if not path:
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["#", "Address", "Value", "Hex (raw)", "Bin (raw)", "Raw"])
                for iid in self.tree.get_children():
                    row = self.tree.item(iid, "values")
                    w.writerow(row)
            messagebox.showinfo("Save CSV", f"Saved: {path}")
            self.log(f"Results CSV saved to {path}", "INFO")
        except Exception as e:
            self.log(f"Save CSV error: {e}", "ERROR")
            show_error("Save CSV error", e)

    # ---------- Connection dot ----------
    def _update_conn_dot(self, connected: bool):
        if not self.conn_dot or not self.conn_dot_item:
            return
        color_fill = "#00a000" if connected else "#c00000"
        color_outline = "#008000" if connected else "#a00000"
        try:
            self.conn_dot.itemconfig(self.conn_dot_item, fill=color_fill, outline=color_outline)
        except Exception:
            pass

    # ---------- Settings save/load helpers ----------
    def _gather_settings(self) -> dict:
        return {
            "conn": {
                "mode": self.conn_mode.get(),
                "unit_id": self.unit_id_var.get(),
                "timeout": self.timeout_var.get(),
                "tcp": {
                    "host": self.tcp_host_var.get(),
                    "port": self.tcp_port_var.get(),
                },
                "rtu": {
                    "port": normalize_device_from_label(self.rtu_port_var.get()),
                    "baud": self.rtu_baud_var.get(),
                    "bytesize": self.rtu_bytesize_var.get(),
                    "parity": self.rtu_parity_var.get(),
                    "stopbits": self.rtu_stopbits_var.get(),
                },
            },
            "read": {
                "fc": self.read_fc_var.get(),
                "addr": self.read_addr_var.get(),
                "count": self.read_count_var.get(),
                "decode_type": self.decode_type_var.get(),
                "byte_order": self.byte_order_var.get(),
                "word_order": self.word_order_var.get(),
            },
            "write": {
                "target": self.write_target_var.get(),
                "addr": self.write_addr_var.get(),
                "values": self.write_values_var.get(),
                "encode_type": self.encode_type_var.get(),
                "byte_order": self.w_byte_order_var.get(),
                "word_order": self.w_word_order_var.get(),
            },
            "poll": {
                "enabled": bool(self.polling),
                "interval": self.poll_interval_var.get(),
            },
            "log": {
                "filter": self.log_filter_var.get(),
                "max_lines": self.log_max_lines_var.get()
            },
            "display": {
                "num_base": self.num_base_var.get(),
                "bin_group": self.bin_group_var.get()
            },
            "window": {
                "geometry": self.root.winfo_geometry()
            }
        }

    def _save_settings(self):
        settings = self._gather_settings()
        save_settings(settings)

    def _save_settings_lazy(self):
        self._save_settings()

    def on_close(self):
        if self.polling:
            self._stop_polling()
        self._save_settings()
        self.log("Application closed.", "INFO")
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ModbusApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()