# Visual Studio and WSL development

## Useful links
Video to install Linux and embedded systems package for Visual Studio and start developing:\
https://www.youtube.com/watch?v=77w9UB_auN8&ab_channel=MicrosoftVisualStudio
\
https://learn.microsoft.com/en-us/cpp/build/walkthrough-build-debug-wsl2?view=msvc-170

Connecting and routing USB devices to WSL:\
https://learn.microsoft.com/en-us/windows/wsl/connect-usb#attach-a-usb-device \
Warning: every device needs to be attached again after disconnecting!

## Useful commands
Listing all serial ports in WSL:\
dmesg | grep tty

# Linux and serial ports

Using serial ports in Linux:
https://blog.mbedded.ninja/programming/operating-systems/linux/linux-serial-ports-using-c-cpp/

# Sharing USB Ports with WSL on Windows

This guide explains how to find unused USB ports and share USB devices with Windows Subsystem for Linux (WSL) distributions using the `usbipd` tool.

## Prerequisites

- **Windows 10, version 21H2 or later** or **Windows 11**.
- **WSL 2** with your preferred distribution installed (e.g., Ubuntu, Arch Linux).
- **`usbipd-win`** tool installed for USB support in WSL.

### Installing `usbipd-win`

1. **Download** the latest `.msi` installer from the [usbipd-win GitHub Releases page](https://github.com/dorssel/usbipd-win/releases).
2. **Run** the installer and follow the prompts to complete the installation.
3. **Verify** the installation by opening **PowerShell** (as Administrator) and running:
   ```powershell
   usbipd --version

### Linking USB port to WSL
1. Check for Available USB Devices
     a) Run the following command to list connected USB devices:
   ```powershell
   usbipd list
  Example output:
  BUSID  VID:PID    DEVICE                                    STATE
  1-6    5986:211b  HD Webcam                                 Not shared
  1-10   8087:0026  Intel(R) Wireless Bluetooth(R)            Not shared
  2-2    046d:c08b  G502 HERO, USB Input Device               Not shared
  3-1    1532:021e  Razer Ornata Chroma, USB Input Device     Not shared


3. Attach a USB Device to a WSL Distribution
    a) To share a specific USB device with WSL, identify its BUSID from the output of usbipd list and use the attach command.
   ```powershell
   usbipd attach --busid <BUSID>
   or
   usbipd attach --busid <BUSID> --distribution <DistroName>
  
NOTE:
Replace <BUSID> with the actual BUSID of the device you want to attach, such as 3-1 for the Razer Ornata Chroma.

5. Verify the Device in WSL
    a) Open your WSL distribution.
    b) Check for the USB device using the lsusb command
   ```powershell
   lsusb

7. Detach the USB Device When Finished
To release the USB device from WSL, run:
   ```powershell
   usbipd detach --busid <BUSID>

### Documentation of UART packages
https://dev.ti.com/tirex/explore/node?node=A__ADnbI7zK9bSRgZqeAxprvQ__com.ti.mmwave_industrial_toolbox__VLyFKFf__4.12
\
https://dev.ti.com/tirex/explore/node?node=A__ADnbI7zK9bSRgZqeAxprvQ__radar_toolbox__1AslXXD__LATEST
