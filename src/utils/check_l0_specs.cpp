#include <iostream>
#include <vector>
#include <level_zero/ze_api.h>

#define ZE_CHECK(res) if(res != ZE_RESULT_SUCCESS) { std::cerr << "Level Zero Error: 0x" << std::hex << (int)res << std::dec << std::endl; return; }

void check_level_zero_specs() {
    ZE_CHECK(zeInit(0));

    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);
    std::vector<ze_driver_handle_t> drivers(driverCount);
    zeDriverGet(&driverCount, drivers.data());

    for (auto driver : drivers) {
        uint32_t deviceCount = 0;
        zeDeviceGet(driver, &deviceCount, nullptr);
        std::vector<ze_device_handle_t> devices(deviceCount);
        zeDeviceGet(driver, &deviceCount, devices.data());

        for (auto device : devices) {
            ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            zeDeviceGetProperties(device, &props);
            if (props.type != ZE_DEVICE_TYPE_GPU) continue;

            std::cout << "\n========== INTEL LEVEL ZERO DIAGNOSTICS (v1.24) ==========" << std::endl;
            std::cout << "Device Name       : " << props.name << std::endl;
            std::cout << "Vendor ID         : 0x" << std::hex << props.vendorId << std::dec << std::endl;
            std::cout << "Core Clock Rate   : " << props.coreClockRate << " MHz" << std::endl;

            // 1. 偵測 USM 能力
            ze_device_memory_access_properties_t memProps = {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
            zeDeviceGetMemoryAccessProperties(device, &memProps);
            std::cout << "\n--- USM (Unified Shared Memory) ---" << std::endl;
            std::cout << "Host Alloc Caps   : 0x" << std::hex << memProps.hostAllocCapabilities << std::dec << std::endl;
            std::cout << "Device Alloc Caps : 0x" << std::hex << memProps.deviceAllocCapabilities << std::dec << std::endl;

            // 2. 偵測 Compute 能力
            ze_device_compute_properties_t computeProps = {ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
            zeDeviceGetComputeProperties(device, &computeProps);
            std::cout << "\n--- Compute Properties ---" << std::endl;
            std::cout << "Max Group Size X  : " << computeProps.maxGroupSizeX << std::endl;
            std::cout << "Max Shared Local  : " << computeProps.maxSharedLocalMemory / 1024 << " KB" << std::endl;

            std::cout << "==========================================================\n" << std::endl;
        }
    }
}

int main() {
    check_level_zero_specs();
    return 0;
}
