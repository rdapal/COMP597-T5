import pynvml

class GpuInfo:

    def __init__(self):
        self.nb_gpu = pynvml.nvmlDeviceGetCount()
        self.handles = []
        for i in range(self.nb_gpu):
            self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
        self.clock_types = [
            pynvml.NVML_CLOCK_GRAPHICS,
            pynvml.NVML_CLOCK_MEM,
            pynvml.NVML_CLOCK_SM,
            pynvml.NVML_CLOCK_VIDEO,
        ]
        self.clock_types_str = {
            pynvml.NVML_CLOCK_GRAPHICS: "GRAPHICS",
            pynvml.NVML_CLOCK_MEM: "MEM",
            pynvml.NVML_CLOCK_SM: "SM",
            pynvml.NVML_CLOCK_VIDEO: "VIDEO",
        }
        self.clock_types_current_supported = {
            pynvml.NVML_CLOCK_GRAPHICS: True,
            pynvml.NVML_CLOCK_MEM: True,
            pynvml.NVML_CLOCK_SM: True,
            pynvml.NVML_CLOCK_VIDEO: True,
        }
        self.clock_types_app_clock_default_supported = {
            pynvml.NVML_CLOCK_GRAPHICS: True,
            pynvml.NVML_CLOCK_MEM: True,
            pynvml.NVML_CLOCK_SM: True,
            pynvml.NVML_CLOCK_VIDEO: False,
        }
        self.clock_types_app_clock_target_supported = {
            pynvml.NVML_CLOCK_GRAPHICS: True,
            pynvml.NVML_CLOCK_MEM: True,
            pynvml.NVML_CLOCK_SM: True,
            pynvml.NVML_CLOCK_VIDEO: False,
        }
        self.clock_types_customer_boost_max_supported = {
            pynvml.NVML_CLOCK_GRAPHICS: True,
            pynvml.NVML_CLOCK_MEM: False,
            pynvml.NVML_CLOCK_SM: True,
            pynvml.NVML_CLOCK_VIDEO: False,
        }
        self.support_maps = {
            pynvml.NVML_CLOCK_ID_CURRENT: self.clock_types_current_supported,
            pynvml.NVML_CLOCK_ID_APP_CLOCK_DEFAULT: self.clock_types_app_clock_default_supported,
            pynvml.NVML_CLOCK_ID_APP_CLOCK_TARGET: self.clock_types_app_clock_target_supported,
            pynvml.NVML_CLOCK_ID_CUSTOMER_BOOST_MAX: self.clock_types_customer_boost_max_supported,
        }

    def _get_clock(self, handle, clock_type, id):
        support_map = self.support_maps[id]
        if support_map[clock_type]:
            return pynvml.nvmlDeviceGetClock(handle, clock_type, id)
        else:
            return "N/A"

    def _print_clock_type(self, handle, clock_type):
        index = pynvml.nvmlDeviceGetIndex(handle)
        default = self._get_clock(handle, clock_type, pynvml.NVML_CLOCK_ID_APP_CLOCK_DEFAULT)
        app_target = self._get_clock(handle, clock_type, pynvml.NVML_CLOCK_ID_APP_CLOCK_TARGET)
        current = self._get_clock(handle, clock_type, pynvml.NVML_CLOCK_ID_CURRENT)
        customer_boost_max = self._get_clock(handle, clock_type, pynvml.NVML_CLOCK_ID_CUSTOMER_BOOST_MAX)
        print(f"| {index :<3} | {self.clock_types_str[clock_type] :<10} | {default :>7} | {current :>7} | {app_target :>18} | {customer_boost_max :>18} |")

    def _print_clock(self, handle):
        for clock_type in self.clock_types:
            self._print_clock_type(handle, clock_type)

    def _print_devices_clocks(self):
        for handle in self.handles:
            self._print_clock(handle)
            print("|-----|------------|---------|---------|--------------------|--------------------|")

    def print(self):
        print("| GPU | Clock Type | Default | Current | Application Target | Customer Boost Max |")
        print("|-----|------------|---------|---------|--------------------|--------------------|")
        self._print_devices_clocks()

if __name__ == "__main__":
    pynvml.nvmlInit()
    info = GpuInfo()
    info.print()
