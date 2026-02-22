# Modbus Reader/Writer (GUI)

A small desktop GUI application for interacting with Modbus devices over TCP and RTU serial links. The app provides read/write operations, polling, data decoding (ints/floats), and flexible display modes (Dec/Hex/Both, binary grouping).

**Key Features**
- GUI-based Modbus master supporting both TCP and RTU.
- Read and write registers/coils with selectable function codes.
- Multiple decode/encode types (raw16, int16/32/64, uint16/32/64, float32/64).
- Byte/word ordering controls for multi-word types.
- Polling loop for continuous reads and an interactive log window.
- Settings persisted to a JSON file for convenience.

**Requirements**
- Python 3.13 or newer
- Dependencies (install via pip):
  - `pymodbus` (modern branch, recommended >= 3.10)
  - `pyserial` (for RTU serial support)
  - `tkinter` (part of the Python standard library on most platforms)

Install dependencies example:

```bash
python -m pip install "pymodbus>=3.10" pyserial
```

**Quick Start**
- Run the application:

```bash
python modbus_app.py
```

- The app stores user preferences in [modbus_settings.json](modbus_settings.json). You can preconfigure connection defaults (TCP host/port or RTU COM/baud), read/write defaults, and window geometry in that file. The GUI also saves changes automatically.

Default settings (already in repository): [modbus_settings.json](modbus_settings.json)

**Files of Interest**
- [modbus_app.py](modbus_app.py): Main application source (Tkinter GUI, Modbus client logic, settings persistence).
- [modbus_settings.json](modbus_settings.json): Persisted default settings used by the app.
- [build_script.bat](build_script.bat): Convenience script to build a distributable with Nuitka (project-specific; inspect before running).
- [modbus_app.iss](modbus_app.iss): Inno Setup script for building a Windows installer.
- [nuitka.txt](nuitka.txt): Notes or configuration related to building with Nuitka.

**Configuration Notes**
- Connection: The settings file contains a `conn` section with `mode` set to either `TCP` or `RTU`. For TCP set `tcp.host` and `tcp.port`. For RTU set `rtu.port` (e.g., `COM3`) and `rtu.baud`.
- Unit ID / Timeout: Controlled via `conn.unit_id` and `conn.timeout`.
- Read/Write: Default addresses, counts and decode/encode types are in the `read` and `write` sections.

**Building / Packaging**
- The repo includes a simple `build_script.bat` and an Inno Setup script (`modbus_app.iss`). Review `nuitka.txt` if you plan to use Nuitka to create a standalone binary.

**Safety & Usage**
- This tool performs Modbus master operations against devices on your network or serial bus. Use with care and ensure you have permission to access target devices.

**Support / Contact**
- For issues and enhancements, inspect `modbus_app.py` and open an issue in your tracker.

**License**
- This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---
Generated from repository files: [modbus_app.py](modbus_app.py), [modbus_settings.json](modbus_settings.json), [build_script.bat](build_script.bat), [modbus_app.iss](modbus_app.iss), [nuitka.txt](nuitka.txt).
