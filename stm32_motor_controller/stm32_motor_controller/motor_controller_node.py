#!/usr/bin/env python3
"""
STM32 Motor Controller ROS2 Node
Jetson <-> STM32 UART 통신을 통한 모터 제어
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray, String
import serial
import time

class STM32MotorController(Node):
    def __init__(self):
        super().__init__('stm32_motor_controller')
        
        # 파라미터 선언
        self.declare_parameter('serial_port', '/dev/ttyTHS1')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('timeout', 1.0)
        
        # 파라미터 가져오기
        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        timeout = self.get_parameter('timeout').value
        
        # 시리얼 포트 초기화
        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=timeout
            )
            self.get_logger().info(f'Serial port opened: {port} @ {baud} bps')
            time.sleep(2)  # STM32 초기화 대기
            
            # 시작 메시지 확인
            if self.serial.in_waiting > 0:
                response = self.serial.readline().decode('utf-8').strip()
                self.get_logger().info(f'STM32: {response}')
        
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            self.serial = None
        
        # Subscriber: cmd_vel (속도 명령)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Subscriber: motor_positions (직접 위치 명령)
        self.motor_pos_sub = self.create_subscription(
            Int32MultiArray,
