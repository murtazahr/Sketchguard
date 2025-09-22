#!/usr/bin/env python3
"""
Federated Learning Real-Time Monitor GUI
A modern, dark-themed GUI for monitoring decentralized federated learning workflows.
"""

import sys
import json
import time
import math
import random
import socket
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QGridLayout, QGroupBox,
    QSplitter, QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QPointF, QRectF
from PyQt6.QtGui import (
    QPalette, QColor, QFont, QPainter, QPen, QBrush,
    QRadialGradient, QLinearGradient, QPainterPath
)

import pyqtgraph as pg
import numpy as np

# Configure pyqtgraph for dark theme
pg.setConfigOptions(antialias=True)


@dataclass
class NodeUpdate:
    """Data structure for node status updates."""
    node_id: int
    round: int
    phase: str  # 'training', 'aggregating', 'idle', 'communicating'
    test_accuracy: float
    test_loss: float
    train_samples: int
    neighbors: List[int]
    is_compromised: bool = False
    acceptance_rate: Optional[float] = None
    timestamp: float = 0.0


@dataclass
class NetworkConfig:
    """Network configuration data."""
    num_nodes: int
    graph_type: str
    edges: List[Tuple[int, int]]
    algorithm: str
    num_rounds: int
    attack_percentage: float = 0.0
    compromised_nodes: List[int] = None


class NetworkVisualization(QWidget):
    """Custom widget for visualizing the network topology."""

    def __init__(self):
        super().__init__()
        self.nodes: Dict[int, Tuple[float, float]] = {}  # node_id -> (x, y)
        self.edges: List[Tuple[int, int]] = []
        self.node_states: Dict[int, str] = {}  # node_id -> phase
        self.node_accuracies: Dict[int, float] = {}
        self.compromised_nodes: set = set()
        self.selected_node = None
        self.setMinimumSize(600, 600)
        self.node_radius = 25

    def set_network(self, config: NetworkConfig):
        """Initialize network topology."""
        self.edges = config.edges
        self.compromised_nodes = set(config.compromised_nodes or [])

        # Store number of nodes
        self.num_nodes = config.num_nodes

        # Calculate node positions using force-directed layout
        self._calculate_layout(config.num_nodes)

        # Initialize states
        for i in range(config.num_nodes):
            self.node_states[i] = 'idle'
            self.node_accuracies[i] = 0.0

        self.update()

    def _calculate_layout(self, num_nodes: int):
        """Calculate node positions using circular or force-directed layout."""
        # Use fixed size if widget not yet shown
        width = max(600, self.width())
        height = max(600, self.height())
        center_x, center_y = width / 2, height / 2
        radius = min(center_x, center_y) * 0.7

        # Use circular layout for better visibility
        for i in range(num_nodes):
            angle = 2 * math.pi * i / num_nodes - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.nodes[i] = (x, y)

    def update_node_state(self, node_id: int, phase: str, accuracy: float = None):
        """Update individual node state."""
        self.node_states[node_id] = phase
        if accuracy is not None:
            self.node_accuracies[node_id] = accuracy
        self.update()

    def paintEvent(self, event):
        """Custom painting for network visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark background
        painter.fillRect(self.rect(), QColor(25, 25, 35))

        # Draw grid for visual reference
        painter.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(0, self.width(), 50):
            painter.drawLine(i, 0, i, self.height())
        for i in range(0, self.height(), 50):
            painter.drawLine(0, i, self.width(), i)

        # Draw edges
        edge_pen = QPen(QColor(80, 80, 100), 2)
        painter.setPen(edge_pen)
        for n1, n2 in self.edges:
            if n1 in self.nodes and n2 in self.nodes:
                x1, y1 = self.nodes[n1]
                x2, y2 = self.nodes[n2]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw nodes
        for node_id, (x, y) in self.nodes.items():
            # Determine node color based on state and type
            if node_id in self.compromised_nodes:
                base_color = QColor(200, 50, 50)  # Red for compromised
            else:
                base_color = QColor(50, 150, 250)  # Blue for honest

            # Modify brightness based on phase
            phase = self.node_states.get(node_id, 'idle')
            if phase == 'training':
                color = base_color.lighter(150)
                border_width = 3
            elif phase == 'aggregating':
                color = QColor(250, 200, 50)  # Yellow for aggregating
                border_width = 4
            elif phase == 'communicating':
                color = QColor(150, 250, 150)  # Green for communicating
                border_width = 3
            else:
                color = base_color
                border_width = 2

            # Create gradient for 3D effect
            gradient = QRadialGradient(QPointF(x, y), self.node_radius)
            gradient.setColorAt(0, color.lighter(120))
            gradient.setColorAt(0.7, color)
            gradient.setColorAt(1, color.darker(150))

            # Draw node shadow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
            painter.drawEllipse(QPointF(x + 3, y + 3),
                              self.node_radius + 2, self.node_radius + 2)

            # Draw node
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(255, 255, 255), border_width))
            painter.drawEllipse(QPointF(x, y), self.node_radius, self.node_radius)

            # Draw node ID
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            font = QFont('Arial', 12, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(QRectF(x - 20, y - 10, 40, 20),
                           Qt.AlignmentFlag.AlignCenter, str(node_id))

            # Draw accuracy below node
            if node_id in self.node_accuracies:
                acc = self.node_accuracies[node_id]
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                font = QFont('Arial', 9)
                painter.setFont(font)
                painter.drawText(QRectF(x - 30, y + 25, 60, 20),
                               Qt.AlignmentFlag.AlignCenter, f"{acc:.1%}")

        # Draw legend
        self._draw_legend(painter)

    def _draw_legend(self, painter):
        """Draw legend for node states."""
        legend_x, legend_y = 10, 10
        legend_items = [
            ("Idle", QColor(50, 150, 250)),
            ("Training", QColor(50, 150, 250).lighter(150)),
            ("Aggregating", QColor(250, 200, 50)),
            ("Communicating", QColor(150, 250, 150)),
            ("Compromised", QColor(200, 50, 50))
        ]

        painter.fillRect(legend_x, legend_y, 150, len(legend_items) * 25 + 10,
                        QColor(30, 30, 40, 200))

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 10 + i * 25

            # Draw color box
            painter.fillRect(legend_x + 10, y_pos, 15, 15, color)

            # Draw label
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.setFont(QFont('Arial', 10))
            painter.drawText(legend_x + 35, y_pos + 12, label)


class NodeStatusPanel(QFrame):
    """Panel showing detailed status for a single node."""

    def __init__(self, node_id: int):
        super().__init__()
        self.node_id = node_id
        self.setFrameStyle(QFrame.Shape.Box)
        self.setMaximumHeight(150)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel(f"Node {node_id}")
        header.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                background-color: #2a2a3a;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        layout.addWidget(header)

        # Status grid
        grid = QGridLayout()

        self.phase_label = QLabel("Idle")
        self.accuracy_label = QLabel("0.0%")
        self.loss_label = QLabel("0.000")
        self.samples_label = QLabel("0")
        self.neighbors_label = QLabel("[]")
        self.acceptance_label = QLabel("N/A")

        labels = [
            ("Phase:", self.phase_label),
            ("Accuracy:", self.accuracy_label),
            ("Loss:", self.loss_label),
            ("Samples:", self.samples_label),
            ("Neighbors:", self.neighbors_label),
            ("Accept Rate:", self.acceptance_label)
        ]

        for i, (name, widget) in enumerate(labels):
            name_label = QLabel(name)
            name_label.setStyleSheet("color: #888888; font-size: 11px;")
            widget.setStyleSheet("color: #ffffff; font-size: 11px;")

            row = i // 2
            col = (i % 2) * 2
            grid.addWidget(name_label, row, col)
            grid.addWidget(widget, row, col + 1)

        layout.addLayout(grid)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(10)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.setLayout(layout)
        self.update_status(NodeUpdate(node_id, 0, "idle", 0, 0, 0, []))

    def update_status(self, update: NodeUpdate):
        """Update node status display."""
        # Color coding for phase
        phase_colors = {
            'idle': '#666666',
            'training': '#4a9eff',
            'aggregating': '#ffc107',
            'communicating': '#4caf50'
        }

        color = phase_colors.get(update.phase, '#ffffff')
        self.phase_label.setText(update.phase.title())
        self.phase_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        self.accuracy_label.setText(f"{update.test_accuracy:.2%}")
        self.loss_label.setText(f"{update.test_loss:.3f}")
        self.samples_label.setText(str(update.train_samples))

        # Format neighbors list
        if len(update.neighbors) > 3:
            neighbors_str = f"{update.neighbors[:3]}..."
        else:
            neighbors_str = str(update.neighbors)
        self.neighbors_label.setText(neighbors_str)

        if update.acceptance_rate is not None:
            self.acceptance_label.setText(f"{update.acceptance_rate:.2%}")

        # Update progress based on phase
        if update.phase == 'training':
            self.progress.setValue(50)
        elif update.phase in ['aggregating', 'communicating']:
            self.progress.setValue(75)
        else:
            self.progress.setValue(0)


class MetricsGraphWidget(QWidget):
    """Widget containing accuracy and loss graphs."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget for different metrics
        self.tabs = QTabWidget()

        # Accuracy graph
        self.accuracy_widget = pg.PlotWidget(title="Test Accuracy")
        self.accuracy_widget.setLabel('left', 'Accuracy', units='%')
        self.accuracy_widget.setLabel('bottom', 'Round')
        self.accuracy_widget.showGrid(x=True, y=True, alpha=0.3)
        self.accuracy_widget.setBackground('#1a1a2a')
        self.accuracy_curves = {}

        # Loss graph
        self.loss_widget = pg.PlotWidget(title="Test Loss")
        self.loss_widget.setLabel('left', 'Loss')
        self.loss_widget.setLabel('bottom', 'Round')
        self.loss_widget.showGrid(x=True, y=True, alpha=0.3)
        self.loss_widget.setBackground('#1a1a2a')
        self.loss_curves = {}

        # Acceptance rate graph (for BALANCE/COARSE/UBAR)
        self.acceptance_widget = pg.PlotWidget(title="Acceptance Rate")
        self.acceptance_widget.setLabel('left', 'Acceptance Rate', units='%')
        self.acceptance_widget.setLabel('bottom', 'Round')
        self.acceptance_widget.showGrid(x=True, y=True, alpha=0.3)
        self.acceptance_widget.setBackground('#1a1a2a')
        self.acceptance_curves = {}

        self.tabs.addTab(self.accuracy_widget, "Accuracy")
        self.tabs.addTab(self.loss_widget, "Loss")
        self.tabs.addTab(self.acceptance_widget, "Acceptance Rate")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # Data storage
        self.accuracy_data = {}
        self.loss_data = {}
        self.acceptance_data = {}
        self.max_points = 100

    def initialize_nodes(self, num_nodes: int, compromised_nodes: List[int] = None):
        """Initialize curves for all nodes."""
        # Clear existing plots first
        self.accuracy_widget.clear()
        self.loss_widget.clear()
        self.acceptance_widget.clear()

        # Clear existing data structures
        self.accuracy_data.clear()
        self.loss_data.clear()
        self.acceptance_data.clear()
        self.accuracy_curves.clear()
        self.loss_curves.clear()
        self.acceptance_curves.clear()

        # Clear average data if it exists
        if hasattr(self, 'avg_acc_data'):
            self.avg_acc_data.clear()
        if hasattr(self, 'avg_loss_data'):
            self.avg_loss_data.clear()

        compromised = set(compromised_nodes or [])

        # Color palette
        colors = [
            (255, 100, 100) if i in compromised else (100, 150, 255)
            for i in range(num_nodes)
        ]

        for i in range(num_nodes):
            # Initialize data storage
            self.accuracy_data[i] = deque(maxlen=self.max_points)
            self.loss_data[i] = deque(maxlen=self.max_points)
            self.acceptance_data[i] = deque(maxlen=self.max_points)

            # Create curves
            pen = pg.mkPen(color=colors[i], width=2)

            self.accuracy_curves[i] = self.accuracy_widget.plot(
                pen=pen, name=f"Node {i}"
            )
            self.loss_curves[i] = self.loss_widget.plot(
                pen=pen, name=f"Node {i}"
            )
            self.acceptance_curves[i] = self.acceptance_widget.plot(
                pen=pen, name=f"Node {i}"
            )

        # Add average line
        avg_pen = pg.mkPen(color=(255, 255, 100), width=3, style=Qt.PenStyle.DashLine)
        self.accuracy_avg_curve = self.accuracy_widget.plot(pen=avg_pen, name="Average")
        self.loss_avg_curve = self.loss_widget.plot(pen=avg_pen, name="Average")

    def update_metrics(self, round_num: int, updates: List[NodeUpdate]):
        """Update graphs with new round data."""
        acc_values = []
        loss_values = []

        for update in updates:
            node_id = update.node_id

            # Add data points
            self.accuracy_data[node_id].append((round_num, update.test_accuracy * 100))
            self.loss_data[node_id].append((round_num, update.test_loss))

            if update.acceptance_rate is not None:
                self.acceptance_data[node_id].append((round_num, update.acceptance_rate * 100))

            # Update curves
            if self.accuracy_data[node_id]:
                x_data = [pt[0] for pt in self.accuracy_data[node_id]]
                y_data = [pt[1] for pt in self.accuracy_data[node_id]]
                self.accuracy_curves[node_id].setData(x_data, y_data)
                acc_values.append(update.test_accuracy * 100)

            if self.loss_data[node_id]:
                x_data = [pt[0] for pt in self.loss_data[node_id]]
                y_data = [pt[1] for pt in self.loss_data[node_id]]
                self.loss_curves[node_id].setData(x_data, y_data)
                loss_values.append(update.test_loss)

            if self.acceptance_data[node_id]:
                x_data = [pt[0] for pt in self.acceptance_data[node_id]]
                y_data = [pt[1] for pt in self.acceptance_data[node_id]]
                self.acceptance_curves[node_id].setData(x_data, y_data)

        # Update average curves
        if acc_values:
            avg_acc = np.mean(acc_values)
            if not hasattr(self, 'avg_acc_data'):
                self.avg_acc_data = deque(maxlen=self.max_points)
            self.avg_acc_data.append((round_num, avg_acc))
            x_data = [pt[0] for pt in self.avg_acc_data]
            y_data = [pt[1] for pt in self.avg_acc_data]
            self.accuracy_avg_curve.setData(x_data, y_data)

        if loss_values:
            avg_loss = np.mean(loss_values)
            if not hasattr(self, 'avg_loss_data'):
                self.avg_loss_data = deque(maxlen=self.max_points)
            self.avg_loss_data.append((round_num, avg_loss))
            x_data = [pt[0] for pt in self.avg_loss_data]
            y_data = [pt[1] for pt in self.avg_loss_data]
            self.loss_avg_curve.setData(x_data, y_data)


class SimulatorInterface(QThread):
    """Thread for communicating with the simulator."""

    update_received = pyqtSignal(list)  # List[NodeUpdate]
    config_received = pyqtSignal(NetworkConfig)
    log_message = pyqtSignal(str)

    def __init__(self, port: int = 5555):
        super().__init__()
        self.port = port
        self.running = False
        self.server_socket = None

    def run(self):
        """Run the communication server."""
        self.running = True

        try:
            # Try to find an available port if the default is in use
            ports_to_try = [self.port, 5556, 5557, 5558, 5559]

            for port in ports_to_try:
                try:
                    self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.server_socket.bind(('localhost', port))
                    self.server_socket.listen(5)
                    self.server_socket.settimeout(1.0)  # Non-blocking with timeout

                    self.port = port  # Update to the port that worked
                    self.log_message.emit(f"Monitor listening on port {self.port}")
                    break
                except OSError as e:
                    if port == ports_to_try[-1]:
                        raise  # Re-raise if no ports are available
                    continue

            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"DEBUG: Accepted connection from {addr}")  # Debug
                    self.handle_client(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:  # Only log if we're still supposed to be running
                        self.log_message.emit(f"Error accepting connection: {e}")
                        print(f"DEBUG: Error in accept loop: {e}")  # Debug

        except Exception as e:
            self.log_message.emit(f"Server error: {e}")
            self.log_message.emit("Please close any existing GUI instances and restart")
        finally:
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass

    def handle_client(self, client_socket):
        """Handle incoming client connection."""
        try:
            # Set a timeout for receiving data
            client_socket.settimeout(5.0)

            data = b""
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    print(f"DEBUG: Received {len(chunk)} bytes, total: {len(data)}")
                except socket.timeout:
                    print("DEBUG: Socket timeout while receiving data")
                    break

            print(f"DEBUG: Total data received: {len(data)} bytes")

            if data:
                try:
                    message_str = data.decode('utf-8')
                    print(f"DEBUG: Decoded message: {message_str[:200]}...")  # First 200 chars
                    message = json.loads(message_str)
                    print(f"DEBUG: Received message type: {message['type']}")  # Debug print

                    if message['type'] == 'config':
                        config = NetworkConfig(**message['data'])
                        self.config_received.emit(config)
                        print(f"DEBUG: Emitted config with {config.num_nodes} nodes")  # Debug

                    elif message['type'] == 'update':
                        updates = [NodeUpdate(**u) for u in message['data']]
                        self.update_received.emit(updates)
                        print(f"DEBUG: Emitted {len(updates)} updates for round {updates[0].round if updates else 'unknown'}")  # Debug

                    elif message['type'] == 'log':
                        self.log_message.emit(message['data'])
                        print(f"DEBUG: Emitted log: {message['data'][:50]}...")  # Debug

                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}")
                    print(f"DEBUG: Raw data: {data}")
                except Exception as e:
                    print(f"DEBUG: Error processing message: {e}")

        except Exception as e:
            print(f"DEBUG: Error in handle_client: {e}")
            self.log_message.emit(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def stop(self):
        """Stop the communication server."""
        self.running = False


class FLMonitorGUI(QMainWindow):
    """Main GUI window for FL monitoring."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decentralized Federated Learning Monitor")
        self.setGeometry(100, 100, 1600, 900)

        # Apply dark theme
        self.apply_dark_theme()

        # Initialize components
        self.network_viz = NetworkVisualization()
        self.metrics_widget = MetricsGraphWidget()
        self.node_panels = {}
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)

        # Setup UI
        self.setup_ui()

        # Setup simulator interface
        self.simulator_interface = SimulatorInterface()
        self.simulator_interface.update_received.connect(self.handle_update, Qt.ConnectionType.QueuedConnection)
        self.simulator_interface.config_received.connect(self.handle_config, Qt.ConnectionType.QueuedConnection)
        self.simulator_interface.log_message.connect(self.add_log_message, Qt.ConnectionType.QueuedConnection)
        self.simulator_interface.start()

        # Stats tracking
        self.current_round = 0
        self.start_time = time.time()

    def setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Network visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Network Topology"))
        left_layout.addWidget(self.network_viz)
        left_panel.setLayout(left_layout)

        # Middle panel - Node status panels
        middle_panel = QWidget()
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel("Node Status"))

        self.nodes_scroll = QScrollArea()
        self.nodes_container = QWidget()
        self.nodes_grid = QGridLayout()
        self.nodes_container.setLayout(self.nodes_grid)
        self.nodes_scroll.setWidget(self.nodes_container)
        self.nodes_scroll.setWidgetResizable(True)

        middle_layout.addWidget(self.nodes_scroll)
        middle_panel.setLayout(middle_layout)

        # Right panel - Metrics graphs
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Performance Metrics"))
        right_layout.addWidget(self.metrics_widget)
        right_panel.setLayout(right_layout)

        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(middle_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([500, 400, 700])

        main_layout.addWidget(content_splitter, 1)

        # Bottom - Log output
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        central_widget.setLayout(main_layout)

    def create_header(self):
        """Create header with status information."""
        header = QFrame()
        header.setFrameStyle(QFrame.Shape.Box)
        header.setMaximumHeight(80)

        layout = QHBoxLayout()

        # Title
        title = QLabel("FL Monitor")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4a9eff;
            }
        """)
        layout.addWidget(title)

        layout.addStretch()

        # Status indicators
        self.round_label = QLabel("Round: 0/0")
        self.round_label.setStyleSheet("font-size: 16px; color: #ffffff;")
        layout.addWidget(self.round_label)

        layout.addSpacing(20)

        self.algorithm_label = QLabel("Algorithm: N/A")
        self.algorithm_label.setStyleSheet("font-size: 16px; color: #ffffff;")
        layout.addWidget(self.algorithm_label)

        layout.addSpacing(20)

        self.status_label = QLabel("⚫ Waiting")
        self.status_label.setStyleSheet("font-size: 16px; color: #ffcc00;")
        layout.addWidget(self.status_label)

        header.setLayout(layout)
        return header

    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        dark_stylesheet = """
        QMainWindow {
            background-color: #1a1a2a;
        }
        QWidget {
            background-color: #1a1a2a;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QFrame {
            background-color: #2a2a3a;
            border: 1px solid #3a3a4a;
            border-radius: 5px;
        }
        QLabel {
            color: #ffffff;
            font-size: 12px;
        }
        QGroupBox {
            color: #ffffff;
            border: 2px solid #3a3a4a;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QScrollArea {
            background-color: #1a1a2a;
            border: 1px solid #3a3a4a;
        }
        QScrollBar:vertical {
            background-color: #2a2a3a;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #4a4a5a;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #5a5a6a;
        }
        QTextEdit {
            background-color: #0a0a1a;
            color: #00ff00;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            border: 1px solid #3a3a4a;
        }
        QProgressBar {
            background-color: #2a2a3a;
            border: 1px solid #3a3a4a;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4a9eff;
            border-radius: 5px;
        }
        QTabWidget::pane {
            background-color: #1a1a2a;
            border: 1px solid #3a3a4a;
        }
        QTabBar::tab {
            background-color: #2a2a3a;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #3a3a4a;
        }
        QTabBar::tab:hover {
            background-color: #4a4a5a;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def handle_config(self, config: NetworkConfig):
        """Handle network configuration update."""
        print(f"DEBUG: handle_config called with {config.num_nodes} nodes")  # Debug

        # Clear previous simulation data
        self.log_output.clear()
        self.current_round = 0

        self.add_log_message(f"Received network config: {config.num_nodes} nodes, {config.algorithm} algorithm")

        # Update UI elements
        self.algorithm_label.setText(f"Algorithm: {config.algorithm.upper()}")
        self.round_label.setText(f"Round: 0/{config.num_rounds}")

        # Initialize network visualization
        self.network_viz.set_network(config)

        # Initialize node panels
        self.setup_node_panels(config.num_nodes)

        # Initialize metrics graphs
        self.metrics_widget.initialize_nodes(config.num_nodes, config.compromised_nodes)

        # Update status
        self.status_label.setText("🟢 Connected")
        self.status_label.setStyleSheet("font-size: 16px; color: #00ff00;")

        print(f"DEBUG: Config handling complete")  # Debug

    def setup_node_panels(self, num_nodes: int):
        """Setup individual node status panels."""
        # Clear existing panels
        for panel in self.node_panels.values():
            panel.deleteLater()
        self.node_panels.clear()

        # Clear grid layout
        while self.nodes_grid.count():
            item = self.nodes_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create new panels
        cols = 2
        for i in range(num_nodes):
            panel = NodeStatusPanel(i)
            self.node_panels[i] = panel
            row = i // cols
            col = i % cols
            self.nodes_grid.addWidget(panel, row, col)

    def handle_update(self, updates: List[NodeUpdate]):
        """Handle node updates from simulator."""
        print(f"DEBUG: handle_update called with {len(updates)} updates")  # Debug
        if not updates:
            return

        # Update round number
        self.current_round = updates[0].round
        current_text = self.round_label.text()
        if '/' in current_text:
            max_rounds = current_text.split('/')[1]
            self.round_label.setText(f"Round: {self.current_round}/{max_rounds}")
        print(f"DEBUG: Updated round label to {self.current_round}")  # Debug

        # Update each node
        for update in updates:
            # Update network visualization
            self.network_viz.update_node_state(
                update.node_id,
                update.phase,
                update.test_accuracy
            )

            # Update node panel
            if update.node_id in self.node_panels:
                self.node_panels[update.node_id].update_status(update)

        # Update metrics graphs
        self.metrics_widget.update_metrics(self.current_round, updates)

        # Log summary
        avg_acc = np.mean([u.test_accuracy for u in updates])
        self.add_log_message(f"Round {self.current_round}: Avg accuracy = {avg_acc:.2%}")

    def add_log_message(self, message: str):
        """Add message to log output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")

        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Handle window close event."""
        self.simulator_interface.stop()
        self.simulator_interface.wait()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = FLMonitorGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()