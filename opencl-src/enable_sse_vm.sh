#!/bin/bash
cd  $HOME/VirtualBox\ VMs/
VBoxManage setextradata "ubuntu-14.04" VBoxInternal/CPUM/SSE4.1 1
VBoxManage setextradata "ubuntu-14.04" VBoxInternal/CPUM/SSE4.2 1
