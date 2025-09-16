str = '''
FP32 Baseline Performance:
 Average Latency per batch: 1098.44 ms
 Throughput: 14.57 samples/sec
 Accuracy: 0.9094
 Model Size: 255.45 MB
[W610 16:05:58.224934858 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.

IPEX BF16 Performance (with AMX if available):
 Average Latency per batch: 210.93 ms
 Throughput: 75.86 samples/sec
 Accuracy: 0.9062
 Model Size: 127.78 MB
2025-06-10 16:12:03,316 - _logger.py - IPEX - WARNING - [NotSupported]BatchNorm folding failed during the prepare process.

Starting INT8 static quantization calibration using 10 batches...
  Calibrated with batch 5
  Calibrated with batch 10
INT8 calibration finished.
/root/miniconda3/envs/amx_lab/lib/python3.10/site-packages/intel_extension_for_pytorch/quantization/_quantization_state_utils.py:452: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  args, scale.item(), zp.item(), dtype
/root/miniconda3/envs/amx_lab/lib/python3.10/site-packages/intel_extension_for_pytorch/quantization/_quantization_state.py:491: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if scale.numel() > 1:
INT8 model traced and frozen.
/root/miniconda3/envs/amx_lab/lib/python3.10/site-packages/torch/amp/autocast_mode.py:283: UserWarning: In CPU autocast, but the target dtype is not supported. Disabling autocast.
CPU Autocast only supports dtype of torch.bfloat16, torch.float16 currently.
  warnings.warn(error_message)

IPEX INT8 Static Quantization Performance (with AMX if available):
 Average Latency per batch: 159.19 ms
 Throughput: 100.51 samples/sec
 Accuracy: 0.9031
 Model Size: 132.63 MB
'''
# print(str)



