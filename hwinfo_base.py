import struct
import time
from multiprocessing import shared_memory
from construct import Struct, Int32un, Long

# Connect to HWiNFO shared memory
memory = shared_memory.SharedMemory('Global\\HWiNFO_SENS_SM2')

# Sensor section header struct
sensor_element_struct = Struct(
    'dwSignature' / Int32un,
    'dwVersion' / Int32un,
    'dwRevision' / Int32un,
    'poll_time' / Long,
    'dwOffsetOfSensorSection' / Int32un,
    'dwSizeOfSensorElement' / Int32un,
    'dwNumSensorElements' / Int32un,
    'dwOffsetOfReadingSection' / Int32un,
    'dwSizeOfReadingElement' / Int32un,
    'dwNumReadingElements' / Int32un,
)

sensor_element = sensor_element_struct.parse(memory.buf[0:sensor_element_struct.sizeof()])

# Define reading struct
fmt = '=III128s128s16sdddd'
reading_element_struct = struct.Struct(fmt)

offset = sensor_element.dwOffsetOfReadingSection
length = sensor_element.dwSizeOfReadingElement
target_label = "CPU Package Power"

try:
    while True:
        found = False
        for index in range(sensor_element.dwNumReadingElements):
            start = offset + index * length
            end = start + reading_element_struct.size
            reading = reading_element_struct.unpack(memory.buf[start:end])

            label_orig = reading[3].replace(b'\x00', b'').decode('utf-8', errors='ignore')
            unit = reading[5].replace(b'\x00', b'').decode('mbcs', errors='ignore')

            if label_orig.strip() == target_label and unit == "W":
                value = reading[6]
                print(f"CPU Package Power: {value:.2f} W")
                found = True
                break

        if not found:
            print("CPU Package Power not found.")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")
finally:
    memory.close()
