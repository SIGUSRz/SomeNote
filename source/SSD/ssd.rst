.. include:: ../_static/.special.rst
##############
SSD Firmware
##############

.. contents:: Table of Contents
   :depth: 2

专用名词缩写 (Acronyms)
***************************

:problem:`Structure`
====================

.. note::
    
    * `PCIe`: Peripheral Component Interconnect Express 计算机扩展总线标准
    * `NVMe`: Non-Volatile Memory Host Controller Interface Specification 非易失性存储器传输层协议, 使用PCIe通道, 一般针对固态硬盘
    * `NAND Flash`: 使用NAND逻辑门的闪存, 一般用于存储器中, 如固态硬盘
    * `ASIC`: Application-specific Integrated Circuit 专用集成电路

      * 一般简化为包含PCIe和NAND闪存之间所有部分的结构

    * `HIM`: Hardware Interface Module: 指代直接传递管理PCIe硬件层信息的模块

      * `HABM`: Host Automated Buffer Manager
      * `HA`: Hardware Accelerator
      * `HAWA`: Hardware Accelerator for Write Accumulation
      * `FIM`: Flash Interface Module
      * `LLFS`: Low Level Flash Sequencer
      * `CAP`: Command Automation Processor

    * `FE`: Front-End

:problem:`Utility`
==================

.. note::

    * `BW`: bandwidth 带宽, 总线数据传输能力
    * `GC`: garbage collection
    * `WL`: wear leveling
    * `OP`: overprovisioning
    * `EI`: error injection

.. _link-pcie:

PCIe (Peripheral Component Interconnect Express)
************************************************

Definition
==========

.. note::

    * 计算机扩展总线标准
    * 管理 Peripheral Devices (外部设备, 即计算机扩展) 之间互联的数据传输总线
    * Serial (串行): 串行信号提高抗干扰性, 增加带宽
    * Point-to-point: 点对点
    * 2 unidirectional differential signal pairs: 差分信号双通道, 收取 + 传输数据
    * Packet-based

Structure
=========

.. note::
    
    * 物理层: 物理接口规格
    * 数据链路层: Transaction Layer, packet based

NVMe (Non-Volatile Memory Host Controller Interface Specification (NVMHCIS))

.. note::
    
    * Define interaction between Host and SSD using `PCIe` interface
    * Host place commands in `submission queue`
    * Controller place completed commands in `completion queue`
    * SSD device access queues through PCIe inside host memory