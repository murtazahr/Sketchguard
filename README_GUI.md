# Federated Learning Real-Time Monitor GUI

A modern, dark-themed GUI for monitoring decentralized federated learning workflows in real-time.

## Features

### 🎨 Modern Dark Theme Interface
- Professional dark color scheme optimized for extended viewing
- Smooth animations and transitions
- Responsive layout that adapts to different screen sizes

### 📊 Real-Time Visualizations

#### Network Topology Visualization
- Interactive network graph showing all nodes and their connections
- Color-coded nodes:
  - Blue: Honest nodes
  - Red: Compromised/Byzantine nodes
- Dynamic state indicators:
  - Idle: Default color
  - Training: Bright blue glow
  - Aggregating: Yellow
  - Communicating: Green
- Real-time accuracy display below each node
- Visual legend for easy reference

#### Performance Metrics Graphs
- **Accuracy Graph**: Track test accuracy for all nodes over rounds
- **Loss Graph**: Monitor training loss evolution
- **Acceptance Rate Graph**: View neighbor acceptance rates (for BALANCE/COARSE/UBAR)
- Individual node curves with color coding
- Average performance line (dashed yellow)
- Smooth real-time updates

#### Node Status Panels
- Individual panels for each node showing:
  - Current phase (Training/Aggregating/Idle)
  - Test accuracy percentage
  - Loss value
  - Number of training samples
  - Neighbor list
  - Acceptance rate (for robust algorithms)
- Progress bars indicating activity
- Color-coded status indicators

### 🔄 Communication Protocol
- Socket-based communication between simulator and GUI
- JSON message protocol for reliable data transfer
- Automatic reconnection handling
- Low-latency updates

## Installation

```bash
# Install GUI dependencies
pip install -r requirements.txt
```

## Usage

### Method 1: Using Launch Script (Recommended)

```bash
# Basic usage with BALANCE algorithm
python launch_fl_monitor.py --dataset femnist --num-nodes 8 --rounds 20 --agg balance

# With COARSE algorithm
python launch_fl_monitor.py --dataset femnist --num-nodes 8 --rounds 20 --agg coarse --coarse-sketch-size 1000

# With UBAR algorithm
python launch_fl_monitor.py --dataset femnist --num-nodes 8 --rounds 20 --agg ubar --ubar-rho 0.4

# With attack simulation
python launch_fl_monitor.py --dataset femnist --num-nodes 8 --rounds 20 --agg balance \
    --attack-percentage 0.25 --attack-type gaussian

# Different network topologies
python launch_fl_monitor.py --dataset femnist --num-nodes 10 --rounds 30 --agg balance \
    --graph k-regular --k 4

# Run simulator without GUI
python launch_fl_monitor.py --dataset femnist --num-nodes 8 --rounds 20 --agg balance --no-gui
```

### Method 2: Manual Launch

```bash
# Terminal 1: Start the GUI monitor
python fl_monitor_gui.py

# Terminal 2: Run the simulator with GUI integration
python decentralized_fl_sim_with_gui.py --dataset femnist --num-nodes 8 --rounds 20 --agg balance --gui
```

## Architecture

### Components

1. **fl_monitor_gui.py**: Main GUI application
   - `FLMonitorGUI`: Main window and UI management
   - `NetworkVisualization`: Custom widget for network topology
   - `NodeStatusPanel`: Individual node status displays
   - `MetricsGraphWidget`: Real-time performance graphs
   - `SimulatorInterface`: Socket server for receiving updates

2. **decentralized_fl_sim_with_gui.py**: Modified simulator with GUI support
   - `GUIConnector`: Handles communication with GUI
   - Sends configuration, updates, and log messages
   - Maintains real-time synchronization

3. **launch_fl_monitor.py**: Convenience launcher
   - Starts both GUI and simulator
   - Manages process lifecycle
   - Handles graceful shutdown

### Communication Protocol

Messages are JSON-encoded with the following types:

#### Configuration Message
```json
{
  "type": "config",
  "data": {
    "num_nodes": 8,
    "graph_type": "ring",
    "edges": [[0,1], [1,2], ...],
    "algorithm": "balance",
    "num_rounds": 20,
    "attack_percentage": 0.0,
    "compromised_nodes": []
  }
}
```

#### Update Message
```json
{
  "type": "update",
  "data": [
    {
      "node_id": 0,
      "round": 5,
      "phase": "training",
      "test_accuracy": 0.85,
      "test_loss": 0.45,
      "train_samples": 1000,
      "neighbors": [1, 7],
      "is_compromised": false,
      "acceptance_rate": 0.75,
      "timestamp": 1234567890.123
    }
  ]
}
```

#### Log Message
```json
{
  "type": "log",
  "data": "Round 5: Starting aggregation"
}
```

## Supported Algorithms

- **D-FedAvg**: Decentralized Federated Averaging
- **Krum**: Byzantine-robust aggregation
- **BALANCE**: Adaptive threshold-based filtering
- **COARSE**: Count-Sketch compression with filtering
- **UBAR**: Two-stage Byzantine-resilient aggregation

## Supported Graphs

- **Ring**: Each node connected to adjacent nodes
- **Fully Connected**: All nodes connected to each other
- **Erdos-Renyi**: Random graph with edge probability
- **K-Regular**: Each node has exactly k neighbors

## Performance Considerations

- The GUI uses PyQt6 for smooth rendering
- Pyqtgraph provides optimized real-time plotting
- Socket communication minimizes latency
- Automatic data pruning prevents memory growth
- Multi-threading keeps UI responsive

## Troubleshooting

### GUI doesn't start
- Ensure PyQt6 is installed: `pip install PyQt6 pyqtgraph`
- Check Python version (3.7+ required)

### Connection issues
- Verify port 5555 is available
- Check firewall settings
- Try restarting both GUI and simulator

### Performance issues
- Reduce number of nodes for smoother visualization
- Close other applications
- Use smaller datasets for testing

## Customization

### Change colors/theme
Edit the `apply_dark_theme()` method in `FLMonitorGUI` class.

### Modify port
Change port in both `SimulatorInterface` (GUI) and `GUIConnector` (simulator).

### Add new metrics
1. Add new data structure in `NodeUpdate`
2. Create new graph in `MetricsGraphWidget`
3. Send data from simulator
4. Update display in GUI

## Examples

### Monitor BALANCE with attacks
```bash
python launch_fl_monitor.py \
    --dataset femnist \
    --num-nodes 10 \
    --rounds 30 \
    --agg balance \
    --balance-gamma 2.0 \
    --balance-kappa 1.0 \
    --attack-percentage 0.3 \
    --attack-type gaussian \
    --graph k-regular \
    --k 4
```

### Compare algorithms
Run multiple instances with different algorithms to compare performance visually.

## Screenshots

The GUI features:
- Dark professional theme
- Network topology with node states
- Real-time accuracy/loss graphs
- Individual node status panels
- System log for debugging

## License

Same as the main project.