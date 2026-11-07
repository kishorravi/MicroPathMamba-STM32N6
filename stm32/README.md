# STM32N6 deployment and on-board measurement (Tier B and Tier C)

Hardware: **STM32N6570-DK** (Cortex-M55 @ 800 MHz, Neural-ART NPU, 4.2 MB internal RAM,
external octo-SPI flash and hexa-SPI PSRAM). Software: **ST Edge AI Core** (CLI `stedgeai`),
STM32CubeIDE, optionally the **STM32 AI Model Zoo services** (image-classification deployment
for STM32N6570-DK). Command flags change between ST Edge AI Core releases, so confirm each
one with `stedgeai --help` and the release notes of the version you install. Record that
version number in the paper.

## 1. Tier B: can each model be deployed?

Run the analysis on every exported model: the student and each teacher/baseline, in FP32 and INT8.

```bash
stedgeai analyze --model deploy/student_int8.onnx  --target stm32n6 --st-neural-art
stedgeai analyze --model deploy/teachers/mambavision_t_int8.onnx --target stm32n6 --st-neural-art
stedgeai analyze --model deploy/baselines/mobilenetv3_s_int8.onnx --target stm32n6 --st-neural-art
```

From each report, copy these values into **Table: Deployability** in the paper:

| column | where in the report |
|---|---|
| Analysis status | OK / error message (e.g. unsupported operator, out of memory) |
| Weights (flash) | "weights (ro)" |
| Activations (RAM) | "activations (rw)" |
| Fits internal RAM? | activations + any internal weights <= 4.2 MB |
| % epochs on NPU (HW) vs CPU (SW) | Neural-ART epoch list: count HW vs SW epochs |
| Unsupported / fallback operators | the operator names the report flags |
| Estimated cycles / latency | if reported |

You should expect the following outcome, but report what the tool actually says:

- **MobileNetV3-S and MicroPathMamba** should fit and run mostly on the NPU.
- **MambaVision-T and VMamba-T** need about 30 MB of INT8 weights. They will need external memory or fail. Their unrolled scan gives thousands of nodes, much of it CPU fallback.

## 2. Tier C: build, flash and validate the student on the board

```bash
# generate the C code for the NPU
stedgeai generate --model deploy/student_int8.onnx --target stm32n6 --st-neural-art --output stm32/gen
# validate on target: runs the same inputs on the board and on the host, compares outputs
stedgeai validate --model deploy/student_int8.onnx --target stm32n6 --mode target -d serial:921600
```

Report the following:

- **Latency.** Use the on-target inference time from `validate`: mean, SD and p95 over at least 100 runs.
- **Board agreement.** Give the top-1 agreement between board and host INT8 predictions on 1 000 CRC-VAL-HE-7K patches. Use the Model Zoo validation app, or stream patches over UART.
- **Accuracy.** Report CRC-VAL-HE-7K accuracy, macro-F1 and MCC at three points:
  - FP32 PyTorch;
  - INT8 ONNX on the host (`evaluate.py --onnx`);
  - INT8 on the board, for the subset you streamed.
- **Memory.** Give peak RAM and flash as they appear in the generated report.

## 3. Energy per inference

1. Power the board through **STLINK-V3PWR** or **X-NUCLEO-LPM01A** and log current at ≥ 50 kHz.
2. Firmware loop: idle 1 s, then 100 back-to-back inferences, then idle 1 s. Toggle a GPIO at each inference boundary.
3. Energy per inference = ∫(V·I)dt over the inference window ÷ 100. Also report the idle baseline and the net value.
4. Fix and report the clocks: CPU 800 MHz and the NPU frequency. Note whether weights sit in internal RAM or external flash.

## 4. Fair comparison rules (state them in the paper)

- Use the same input size (224×224×3) and the same INT8 QDQ per-channel scheme for every deployed model.
- Calibrate on 512 training patches from the same fold.
- Use the same compiler version and options.
- Report the baselines (MobileNetV3-S, MobileNetV4-Conv-S and the `micropathmamba_cnn` ablation without SSM blocks) on the same board.
