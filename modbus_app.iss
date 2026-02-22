[Setup]
AppName=Modbus Modbus Reader/Writer
AppVersion=1.0
DefaultDirName={pf}\ModbusTool
DefaultGroupName=Modbus
OutputDir=installer
OutputBaseFilename=ModbusAppInstaller
Compression=lzma
SolidCompression=yes
SetupIconFile=app.ico

[Files]
Source:"build\modbus_app.dist\*"; DestDir:"{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name:"{group}\Modbus Tool"; Filename:"{app}\ModbusTool.exe"
Name:"{commondesktop}\Modbus Tool"; Filename:"{app}\Modbustool.exe"; Tasks:desktopicon

[Tasks]
Name:desktopicon; Description:"Create a desktop shortcut"; Flags:unchecked

[Code]
function InitializeUninstall(): Boolean;
begin
  Result := MsgBox('InitializeUninstall:' #13#13 'Uninstall is initializing. Do you really want to start Uninstall?', mbConfirmation, MB_YESNO) = idYes;
  if Result = False then
    MsgBox('InitializeUninstall:' #13#13 'Ok, bye bye.', mbInformation, MB_OK);
end;

procedure DeinitializeUninstall();
begin
  MsgBox('DeinitializeUninstall:' #13#13 'Bye bye!', mbInformation, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  case CurUninstallStep of
    usUninstall:
      begin
        MsgBox('CurUninstallStepChanged:' #13#13 'Uninstall is about to start.', mbInformation, MB_OK)
        // ...insert code to perform pre-uninstall tasks here...
      end;
    usPostUninstall:
      begin
        MsgBox('CurUninstallStepChanged:' #13#13 'Uninstall just finished.', mbInformation, MB_OK);
        // ...insert code to perform post-uninstall tasks here...
      end;
  end;
end;
