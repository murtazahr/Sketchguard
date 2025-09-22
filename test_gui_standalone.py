#!/usr/bin/env python3
"""
Test the GUI directly without socket communication.
"""

import sys
from PyQt6.QtWidgets import QApplication
from fl_monitor_gui import FLMonitorGUI, NetworkConfig, NodeUpdate
import random


def test_gui():
    """Test GUI with direct method calls."""
    app = QApplication(sys.argv)

    window = FLMonitorGUI()
    window.show()

    # Stop the socket server thread
    window.simulator_interface.stop()

    # Directly call handle_config
    config = NetworkConfig(
        num_nodes=8,
        graph_type="ring",
        edges=[(i, (i+1)%8) for i in range(8)],
        algorithm="balance",
        num_rounds=20,
        attack_percentage=0.25,
        compromised_nodes=[1, 3]
    )

    print("Calling handle_config directly...")
    window.handle_config(config)

    # Directly call handle_update
    updates = []
    for i in range(8):
        update = NodeUpdate(
            node_id=i,
            round=1,
            phase='training',
            test_accuracy=random.uniform(0.1, 0.3),
            test_loss=random.uniform(2.0, 3.0),
            train_samples=1000,
            neighbors=[(i-1)%8, (i+1)%8],
            is_compromised=(i in [1, 3]),
            acceptance_rate=random.uniform(0.7, 1.0)
        )
        updates.append(update)

    print("Calling handle_update directly...")
    window.handle_update(updates)

    window.add_log_message("Test: Direct method calls successful!")

    sys.exit(app.exec())


if __name__ == "__main__":
    print("Testing GUI with direct method calls...")
    test_gui()