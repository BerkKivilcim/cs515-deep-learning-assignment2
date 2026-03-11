# CS515 Homework Solution Skeleton

This repository supports both homework parts:

- **Part A — Transfer Learning**
  - `resize_freeze`: resize CIFAR-10 images to 224×224, load ImageNet pretrained model, freeze backbone, train classifier head.
  - `modify_finetune`: adapt the early convolution/pooling stem for CIFAR-10 sized inputs, initialize from ImageNet weights where possible, and fine-tune the network.

- **Part B — Knowledge Distillation**
  - Train a **SimpleCNN** on CIFAR-10.
  - Train **ResNet-18 from scratch** with and without label smoothing.
  - Distill a trained ResNet teacher into **SimpleCNN**.
  - Distill a trained ResNet teacher into **MobileNetV2** with the custom "true-class probability only" target assignment.
  - Compare FLOPs and parameter counts with `ptflops`.

## Example commands

See the final answer in ChatGPT for the exact sequence of commands to run each experiment.


## Optional graph export

Install `torchviz` and a local Graphviz binary if you would like to export architecture graphs as PNG files. Then run any experiment with `--export_arch_graphs`.
