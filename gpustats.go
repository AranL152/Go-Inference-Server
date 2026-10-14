// GPU readings straight from NVML the library behind nvidia-smi and Task Manager
package main

import (
	"fmt"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

type nvmlGPU struct {
	dev              nvml.Device
	name             string
	pcieGen, pcieW   int
	powerV, thermalV nvml.ViolationTime
}

func openGPU() (*nvmlGPU, error) {
	if r := nvml.Init(); r != nvml.SUCCESS {
		return nil, fmt.Errorf("nvml init: %s", nvml.ErrorString(r))
	}
	dev, r := nvml.DeviceGetHandleByIndex(0)
	if r != nvml.SUCCESS {
		return nil, fmt.Errorf("nvml device 0: %s", nvml.ErrorString(r))
	}
	g := &nvmlGPU{dev: dev}
	g.name, _ = dev.GetName()
	g.pcieGen, _ = dev.GetCurrPcieLinkGeneration()
	g.pcieW, _ = dev.GetCurrPcieLinkWidth()
	g.powerV, _ = dev.GetViolationStatus(nvml.PERF_POLICY_POWER)
	g.thermalV, _ = dev.GetViolationStatus(nvml.PERF_POLICY_THERMAL)
	return g, nil
}

var throttleNames = []struct {
	bit  uint64
	name string
}{
	{nvml.ClocksEventReasonSwPowerCap, "power cap"},
	{nvml.ClocksThrottleReasonHwSlowdown, "hardware slowdown"},
	{nvml.ClocksEventReasonSwThermalSlowdown, "thermal"},
	{nvml.ClocksThrottleReasonHwThermalSlowdown, "hardware thermal"},
	{nvml.ClocksThrottleReasonHwPowerBrakeSlowdown, "power brake"},
	{nvml.ClocksEventReasonBoardLimit, "board limit"},
	{nvml.ClocksEventReasonReliability, "reliability"},
	{nvml.ClocksEventReasonSyncBoost, "sync boost"},
	{nvml.ClocksEventReasonApplicationsClocksSetting, "app clocks"},
	{nvml.ClocksEventReasonDisplayClockSetting, "display clock"},
}

// violationPct is the share of the last interval spent held back by a policy
func violationPct(prev *nvml.ViolationTime, cur nvml.ViolationTime) float64 {
	dRef, dV := cur.ReferenceTime-prev.ReferenceTime, cur.ViolationTime-prev.ViolationTime
	*prev = cur
	if dRef == 0 || cur.ViolationTime < dV {
		return 0
	}
	return min(100, 100*float64(dV)/float64(dRef))
}

func (g *nvmlGPU) sample() *gpuSample {
	d := g.dev
	s := &gpuSample{Name: g.name, PCIeGen: float64(g.pcieGen), PCIeWidth: float64(g.pcieW), Throttle: []string{}}
	if u, r := d.GetUtilizationRates(); r == nvml.SUCCESS {
		s.Util, s.MemUtil = float64(u.Gpu), float64(u.Memory)
	}
	if m, r := d.GetMemoryInfo(); r == nvml.SUCCESS {
		s.MemUsedMB, s.MemTotalMB = float64(m.Used)/(1<<20), float64(m.Total)/(1<<20)
	}
	if t, r := d.GetTemperature(nvml.TEMPERATURE_GPU); r == nvml.SUCCESS {
		s.TempC = float64(t)
	}
	if p, r := d.GetPowerUsage(); r == nvml.SUCCESS {
		s.PowerW = float64(p) / 1000
	}
	if p, r := d.GetEnforcedPowerLimit(); r == nvml.SUCCESS {
		s.PowerMaxW = float64(p) / 1000
	}
	if c, r := d.GetClockInfo(nvml.CLOCK_SM); r == nvml.SUCCESS {
		s.SMClock = float64(c)
	}
	if c, r := d.GetMaxClockInfo(nvml.CLOCK_SM); r == nvml.SUCCESS {
		s.SMClockMax = float64(c)
	}
	if c, r := d.GetClockInfo(nvml.CLOCK_MEM); r == nvml.SUCCESS {
		s.MemClock = float64(c)
	}
	if f, r := d.GetFanSpeed(); r == nvml.SUCCESS {
		s.Fan = float64(f)
	}
	if v, r := d.GetPcieThroughput(nvml.PCIE_UTIL_RX_BYTES); r == nvml.SUCCESS {
		s.PCIeRxMBs = float64(v) / 1024
	}
	if v, r := d.GetPcieThroughput(nvml.PCIE_UTIL_TX_BYTES); r == nvml.SUCCESS {
		s.PCIeTxMBs = float64(v) / 1024
	}
	if v, _, r := d.GetEncoderUtilization(); r == nvml.SUCCESS {
		s.EncUtil = float64(v)
	}
	if v, _, r := d.GetDecoderUtilization(); r == nvml.SUCCESS {
		s.DecUtil = float64(v)
	}
	if v, _, r := d.GetJpgUtilization(); r == nvml.SUCCESS {
		s.JpgUtil = float64(v)
	}
	if bits, r := d.GetCurrentClocksEventReasons(); r == nvml.SUCCESS {
		for _, t := range throttleNames {
			if bits&t.bit != 0 {
				s.Throttle = append(s.Throttle, t.name)
			}
		}
	}
	if v, r := d.GetViolationStatus(nvml.PERF_POLICY_POWER); r == nvml.SUCCESS {
		s.PowerViolPct = violationPct(&g.powerV, v)
	}
	if v, r := d.GetViolationStatus(nvml.PERF_POLICY_THERMAL); r == nvml.SUCCESS {
		s.ThermalViolPct = violationPct(&g.thermalV, v)
	}
	return s
}
