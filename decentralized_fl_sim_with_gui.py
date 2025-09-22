#!/usr/bin/env python3
"""
Modified Decentralized FL Simulator with GUI integration.
This version sends real-time updates to the monitoring GUI.
"""

import socket
import json
import time
from typing import Dict, List, Optional
import sys
import os
from dataclasses import dataclass, asdict

# Import everything from the original simulator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from decentralized_fl_sim import *


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


class GUIConnector:
    """Handles communication with the monitoring GUI."""

    def __init__(self, host: str = 'localhost', port: int = 5555):
        self.host = host
        self.port = port
        self.connected = False

    def connect(self):
        """Attempt to connect to the GUI monitor."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to GUI monitor at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Could not connect to GUI monitor: {e}")
            print("Running without GUI...")
            self.connected = False
            return False

    def send_config(self, config: NetworkConfig):
        """Send network configuration to GUI."""
        if not self.connected:
            return

        try:
            message = {
                'type': 'config',
                'data': asdict(config)
            }
            self._send_message(message)
        except Exception as e:
            print(f"Error sending config: {e}")
            self.connected = False

    def send_updates(self, updates: List[NodeUpdate]):
        """Send node updates to GUI."""
        if not self.connected:
            return

        try:
            message = {
                'type': 'update',
                'data': [asdict(u) for u in updates]
            }
            self._send_message(message)
        except Exception as e:
            print(f"Error sending updates: {e}")
            self.connected = False

    def send_log(self, log_message: str):
        """Send log message to GUI."""
        if not self.connected:
            return

        try:
            message = {
                'type': 'log',
                'data': log_message
            }
            self._send_message(message)
        except Exception as e:
            print(f"Error sending log: {e}")
            self.connected = False

    def _send_message(self, message: dict):
        """Send a JSON message over the socket."""
        if not self.connected:
            return

        try:
            # Create a new connection for each message to ensure delivery
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.connect((self.host, self.port))

            data = json.dumps(message).encode('utf-8')
            temp_socket.sendall(data)
            temp_socket.close()

            print(f"DEBUG: Sent {message['type']} message successfully")
        except Exception as e:
            print(f"DEBUG: Error sending message: {e}")
            self.connected = False

    def close(self):
        """Close the connection."""
        if self.connected:
            self.socket.close()
            self.connected = False


def run_sim_with_gui(args):
    """Modified simulation function with GUI integration."""

    # Initialize GUI connector
    gui = GUIConnector()
    use_gui = gui.connect()

    # Standard initialization
    set_seed(args.seed)
    dev = device()
    print(f"Device: {dev}")
    print(f"Seed: {args.seed}")

    # Load dataset
    if args.dataset.lower() == "femnist":
        data_path = "./leaf/data/femnist/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset("femnist", data_path)
        image_size = input_size
    elif args.dataset.lower() == "celeba":
        data_path = "./leaf/data/celeba/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset("celeba", data_path)
        image_size = input_size
    else:
        raise ValueError(f"Dataset {args.dataset} not supported. Use 'femnist' or 'celeba'")

    # Create client partitions
    train_partitions, test_partitions = create_leaf_client_partitions(train_ds, test_ds, args.num_nodes, seed=args.seed)
    parts = [Subset(train_ds, indices) for indices in train_partitions]
    test_parts = [Subset(test_ds, indices) for indices in test_partitions]

    num_workers = 4
    pin_memory = dev.type != "cpu"

    use_sampling = args.max_samples is not None
    if use_sampling:
        print(f"Will sample {args.max_samples} samples per client per epoch")

    test_loaders = [DataLoader(tp, batch_size=512, shuffle=False,
                               num_workers=0, pin_memory=False) for tp in test_parts]

    # Create graph topology
    graph = make_graph(args.num_nodes, args.graph, p=args.p, k=args.k)
    print(f"Graph: {args.graph}, nodes: {args.num_nodes}, edges: {len(graph.edges)}")

    # Report topology statistics
    degrees = [len(neighbors) for neighbors in graph.neighbors]
    avg_degree = np.mean(degrees)
    min_degree = min(degrees)
    max_degree = max(degrees)
    print(f"Degree statistics: avg={avg_degree:.2f}, min={min_degree}, max={max_degree}")

    # Initialize attacker if requested
    attacker = None
    compromised_nodes = []
    if args.attack_percentage > 0:
        attacker = LocalModelPoisoningAttacker(
            args.num_nodes,
            args.attack_percentage,
            args.attack_type,
            args.attack_lambda,
            args.seed
        )
        compromised_nodes = list(attacker.compromised_nodes)
        print(f"Attack type: {args.attack_type}, lambda: {args.attack_lambda}")

    # Send configuration to GUI
    if use_gui:
        config = NetworkConfig(
            num_nodes=args.num_nodes,
            graph_type=args.graph,
            edges=graph.edges,
            algorithm=args.agg,
            num_rounds=args.rounds,
            attack_percentage=args.attack_percentage,
            compromised_nodes=compromised_nodes
        )
        gui.send_config(config)
        gui.send_log(f"Starting {args.agg.upper()} algorithm with {args.num_nodes} nodes")

    # Initialize models
    models = []
    for i in range(args.num_nodes):
        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + i)

        if args.dataset.lower() == "femnist":
            model = LEAFFEMNISTModel(num_classes=num_classes).to(dev)
        elif args.dataset.lower() == "celeba":
            model = LEAFCelebAModel(num_classes=num_classes, image_size=image_size).to(dev)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if dev.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        models.append(model)

    # Calculate model dimension for sketching algorithms
    model_dim = calculate_model_dimension(models[0])

    # Initialize aggregation monitors
    balance_monitors = {}
    coarse_monitors = {}
    ubar_monitors = {}

    if args.agg == "balance":
        balance_config = BALANCEConfig(
            gamma=args.balance_gamma,
            kappa=args.balance_kappa,
            alpha=args.balance_alpha
        )
        for i in range(args.num_nodes):
            balance_monitors[str(i)] = BALANCE(str(i), balance_config, args.rounds)
        print(f"BALANCE algorithm initialized")

    elif args.agg == "coarse":
        coarse_config = COARSEConfig(
            gamma=args.balance_gamma,
            kappa=args.balance_kappa,
            alpha=args.balance_alpha,
            sketch_size=args.coarse_sketch_size,
            network_seed=args.seed,
            attack_detection_window=5
        )
        for i in range(args.num_nodes):
            coarse_monitors[str(i)] = COARSE(str(i), coarse_config, args.rounds, model_dim)
        print(f"COARSE algorithm initialized")

    elif args.agg == "ubar":
        ubar_config = UBARConfig(
            rho=args.ubar_rho,
            alpha=args.balance_alpha
        )
        # Create training loaders for UBAR
        ubar_train_loaders = []
        for i, p in enumerate(parts):
            subset_size = min(64, len(p))
            subset_indices = random.sample(range(len(p)), subset_size)
            subset_data = Subset(p, subset_indices)
            loader = DataLoader(subset_data, batch_size=32, shuffle=True, num_workers=0)
            ubar_train_loaders.append(loader)

        for i in range(args.num_nodes):
            ubar_monitors[str(i)] = UBAR(str(i), ubar_config, ubar_train_loaders[i], dev)
        print(f"UBAR algorithm initialized")

    # Evaluate initial performance
    with torch.no_grad():
        base_accs = []
        base_losses = []
        for i, m in enumerate(models):
            acc, loss, _, _ = evaluate(m, test_loaders[i], dev)
            base_accs.append(acc)
            base_losses.append(loss)
        print(f"Initial test acc across nodes: mean={np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}")

    # Send initial state to GUI
    if use_gui:
        updates = []
        for i in range(args.num_nodes):
            update = NodeUpdate(
                node_id=i,
                round=0,
                phase='idle',
                test_accuracy=base_accs[i],
                test_loss=base_losses[i],
                train_samples=len(parts[i]),
                neighbors=graph.neighbors[i],
                is_compromised=(i in compromised_nodes),
                acceptance_rate=None,
                timestamp=time.time()
            )
            updates.append(update)
        gui.send_updates(updates)

    # Main training loop
    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} ===")

        # Create data loaders
        if use_sampling:
            loaders = []
            for i, p in enumerate(parts):
                num_samples = min(args.max_samples, len(p))
                round_seed = args.seed + r * 1000 + i
                sampler = RandomSampler(p, replacement=False, num_samples=num_samples,
                                        generator=torch.Generator().manual_seed(round_seed))
                loader = DataLoader(p, batch_size=args.batch_size, sampler=sampler,
                                    num_workers=num_workers, pin_memory=pin_memory)
                loaders.append(loader)
        else:
            loaders = []
            for i, p in enumerate(parts):
                round_seed = args.seed + r * 1000 + i
                generator = torch.Generator().manual_seed(round_seed)
                loader = DataLoader(p, batch_size=args.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    persistent_workers=(num_workers > 0 and r == 1),
                                    prefetch_factor=2 if num_workers > 0 else None,
                                    generator=generator)
                loaders.append(loader)

        # Local training phase - send updates to GUI
        if use_gui:
            updates = []
            for i in range(args.num_nodes):
                update = NodeUpdate(
                    node_id=i,
                    round=r,
                    phase='training',
                    test_accuracy=base_accs[i],  # Previous accuracy
                    test_loss=base_losses[i],     # Previous loss
                    train_samples=len(loaders[i].dataset) if hasattr(loaders[i], 'dataset') else len(parts[i]),
                    neighbors=graph.neighbors[i],
                    is_compromised=(i in compromised_nodes),
                    timestamp=time.time()
                )
                updates.append(update)
            gui.send_updates(updates)
            gui.send_log(f"Round {r}: Starting local training")

        # Perform local training
        for i, (m, ld) in enumerate(zip(models, loaders)):
            local_train(m, ld, epochs=args.local_epochs, lr=args.lr, device_=dev)

        # Communication/aggregation phase - send updates to GUI
        if use_gui:
            updates = []
            for i in range(args.num_nodes):
                phase = 'aggregating' if i not in compromised_nodes else 'communicating'
                update = NodeUpdate(
                    node_id=i,
                    round=r,
                    phase=phase,
                    test_accuracy=base_accs[i],
                    test_loss=base_losses[i],
                    train_samples=len(loaders[i].dataset) if hasattr(loaders[i], 'dataset') else len(parts[i]),
                    neighbors=graph.neighbors[i],
                    is_compromised=(i in compromised_nodes),
                    timestamp=time.time()
                )
                updates.append(update)
            gui.send_updates(updates)
            gui.send_log(f"Round {r}: Starting aggregation ({args.agg.upper()})")

        # Perform aggregation
        if args.agg == "d-fedavg":
            decentralized_fedavg_step(models, graph, r, attacker)
        elif args.agg == "krum":
            decentralized_krum_step(models, graph, args.pct_compromised, r, attacker)
        elif args.agg == "balance":
            balance_aggregation_step(models, graph, balance_monitors, r, attacker)
        elif args.agg == "coarse":
            coarse_aggregation_step(models, graph, coarse_monitors, r, attacker)
        elif args.agg == "ubar":
            ubar_aggregation_step(models, graph, ubar_monitors, r, attacker)
        else:
            raise ValueError("agg must be 'd-fedavg', 'krum', 'balance', 'coarse', or 'ubar'")

        # Evaluation phase
        accs = []
        losses = []
        correct_totals = []
        for i, m in enumerate(models):
            acc, loss, correct, total = evaluate(m, test_loaders[i], dev)
            accs.append(acc)
            losses.append(loss)
            correct_totals.append((correct, total))

        print(f"Round {r:03d}: test acc mean={np.mean(accs):.4f} ± {np.std(accs):.4f} | "
              f"min={np.min(accs):.4f} max={np.max(accs):.4f}")

        # Send evaluation results to GUI
        if use_gui:
            updates = []
            for i in range(args.num_nodes):
                # Get acceptance rate if available
                acceptance_rate = None
                if args.agg == "balance" and str(i) in balance_monitors:
                    if balance_monitors[str(i)].acceptance_history:
                        acceptance_rate = balance_monitors[str(i)].acceptance_history[-1]
                elif args.agg == "coarse" and str(i) in coarse_monitors:
                    if coarse_monitors[str(i)].acceptance_history:
                        acceptance_rate = coarse_monitors[str(i)].acceptance_history[-1]
                elif args.agg == "ubar" and str(i) in ubar_monitors:
                    if ubar_monitors[str(i)].stage2_acceptance_history:
                        acceptance_rate = ubar_monitors[str(i)].stage2_acceptance_history[-1]

                update = NodeUpdate(
                    node_id=i,
                    round=r,
                    phase='idle',
                    test_accuracy=accs[i],
                    test_loss=losses[i],
                    train_samples=len(parts[i]),
                    neighbors=graph.neighbors[i],
                    is_compromised=(i in compromised_nodes),
                    acceptance_rate=acceptance_rate,
                    timestamp=time.time()
                )
                updates.append(update)
            gui.send_updates(updates)

            # Send summary
            avg_acc = np.mean(accs)
            gui.send_log(f"Round {r}: Completed - Avg accuracy: {avg_acc:.2%}")

        # Update base values for next round
        base_accs = accs
        base_losses = losses

    # Final evaluation
    print("\n=== FINAL RESULTS ===")
    print(f"Dataset: {args.dataset}, Nodes: {args.num_nodes}, Graph: {args.graph}, Aggregation: {args.agg}")
    if attacker:
        compromised_accs = [accs[i] for i in attacker.compromised_nodes]
        honest_accs = [accs[i] for i in range(args.num_nodes) if i not in attacker.compromised_nodes]
        if compromised_accs and honest_accs:
            print(f"Attack: {args.attack_type}, {args.attack_percentage*100:.1f}% compromised")
            print(f"Final accuracy - Compromised: {np.mean(compromised_accs):.4f}, Honest: {np.mean(honest_accs):.4f}")
    print(f"Overall test accuracy: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    if use_gui:
        gui.send_log(f"Simulation completed! Final accuracy: {np.mean(accs):.2%}")
        gui.close()


def parse_args_with_gui():
    """Parse command line arguments with GUI support."""
    import argparse

    p = argparse.ArgumentParser(description="Decentralized FL Simulator with GUI Support")

    # Dataset and basic training parameters
    p.add_argument("--dataset", type=str, choices=["femnist", "celeba"], required=True)
    p.add_argument("--num-nodes", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-samples", type=int, default=None)

    # Aggregation algorithm parameters
    p.add_argument("--agg", type=str,
                   choices=["d-fedavg", "krum", "balance", "coarse", "ubar"],
                   default="d-fedavg",
                   help="Aggregation algorithm")
    p.add_argument("--pct-compromised", type=float, default=0.0)

    # BALANCE algorithm parameters
    p.add_argument("--balance-gamma", type=float, default=2.0)
    p.add_argument("--balance-kappa", type=float, default=1.0)
    p.add_argument("--balance-alpha", type=float, default=0.5)

    # COARSE specific parameters
    p.add_argument("--coarse-sketch-size", type=int, default=1000,
                   help="COARSE sketch size k (lower = more compression)")

    # UBAR specific parameters
    p.add_argument("--ubar-rho", type=float, default=0.4,
                   help="UBAR rho parameter (ratio of benign neighbors)")

    # Graph topology parameters
    p.add_argument("--graph", type=str, choices=["ring", "fully", "erdos", "k-regular"], default="ring",
                   help="Graph topology")
    p.add_argument("--p", type=float, default=0.3, help="Edge probability for Erdos-Renyi graphs")
    p.add_argument("--k", type=int, default=4, help="Degree k for k-regular graphs (must be even)")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Attack parameters
    p.add_argument("--attack-percentage", type=float, default=0.0)
    p.add_argument("--attack-type", type=str, choices=["directed_deviation", "gaussian"],
                   default="directed_deviation")
    p.add_argument("--attack-lambda", type=float, default=1.0)

    # Debug/verbose
    p.add_argument("--verbose", action="store_true")

    # GUI support
    p.add_argument("--gui", action="store_true", help="Run with GUI monitor")

    return p.parse_args()


def main():
    """Main entry point."""
    args = parse_args_with_gui()

    # Check if GUI monitor is requested
    if args.gui:
        print("Running with GUI monitor...")
        print("Make sure to start fl_monitor_gui.py first!")
        time.sleep(2)  # Give user time to start GUI if needed

    run_sim_with_gui(args)


if __name__ == "__main__":
    main()