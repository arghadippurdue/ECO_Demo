from HardwareMonitor.Hardware import *  # equivalent to 'using LibreHardwareMonitor.Hardware;'

class UpdateVisitor(IVisitor):
    __namespace__ = "TestHardwareMonitor"  # must be unique among implementations of the IVisitor interface
    def VisitComputer(self, computer: IComputer):
        computer.Traverse(self);

    def VisitHardware(self, hardware: IHardware):
        hardware.Update()
        for subHardware in hardware.SubHardware:
            subHardware.Update()

    def VisitParameter(self, parameter: IParameter): pass

    def VisitSensor(self, sensor: ISensor): pass


computer = Computer()  # settings can not be passed as constructor argument (following below)
computer.IsMotherboardEnabled = True
computer.IsControllerEnabled = True
computer.IsCpuEnabled = True
computer.IsGpuEnabled = True
computer.IsBatteryEnabled = True
computer.IsMemoryEnabled = True
computer.IsNetworkEnabled = True
computer.IsStorageEnabled = True

computer.Open()
computer.Accept(UpdateVisitor())

for hardware in computer.Hardware:
    print(f"Hardware: {hardware.Name}")
    for subhardware  in hardware.SubHardware:
        print(f"\tSubhardware: {subhardware.Name}")
        for sensor in subhardware.Sensors:
            print(f"\t\tSensor: {sensor.Name}, value: {sensor.Value}")
    for sensor in hardware.Sensors:
        print(f"\tSensor: {sensor.Name}, value: {sensor.Value}")

computer.Close()