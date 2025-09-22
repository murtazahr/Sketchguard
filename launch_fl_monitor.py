#!/usr/bin/env python3
"""
Launch script for FL Monitor GUI with simulator.
Starts the GUI monitor and then runs the simulator with GUI integration.
"""

import subprocess
import time
import sys
import os
import argparse
import signal
import socket
from multiprocessing import Process


def run_gui():
    """Run the GUI monitor."""
    try:
        subprocess.run([sys.executable, "fl_monitor_gui.py"], check=True)
    except KeyboardInterrupt:
        pass


def run_simulator(sim_args):
    """Run the simulator with GUI integration."""
    try:
        cmd = [sys.executable, "decentralized_fl_sim_with_gui.py"] + sim_args + ["--gui"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Launch FL Monitor with GUI")

    # Add all simulator arguments
    parser.add_argument("--dataset", type=str, choices=["femnist", "celeba"],
                       default="femnist", help="Dataset to use")
    parser.add_argument("--num-nodes", type=int, default=8, help="Number of nodes")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per client")

    parser.add_argument("--agg", type=str,
                       choices=["d-fedavg", "krum", "balance", "coarse", "ubar"],
                       default="balance", help="Aggregation algorithm")
    parser.add_argument("--pct-compromised", type=float, default=0.0)

    # BALANCE parameters
    parser.add_argument("--balance-gamma", type=float, default=2.0)
    parser.add_argument("--balance-kappa", type=float, default=1.0)
    parser.add_argument("--balance-alpha", type=float, default=0.5)

    # COARSE parameters
    parser.add_argument("--coarse-sketch-size", type=int, default=1000)

    # UBAR parameters
    parser.add_argument("--ubar-rho", type=float, default=0.4)

    # Graph parameters
    parser.add_argument("--graph", type=str,
                       choices=["ring", "fully", "erdos", "k-regular"],
                       default="ring", help="Graph topology")
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=4)

    # Attack parameters
    parser.add_argument("--attack-percentage", type=float, default=0.0)
    parser.add_argument("--attack-type", type=str,
                       choices=["directed_deviation", "gaussian"],
                       default="directed_deviation")
    parser.add_argument("--attack-lambda", type=float, default=1.0)

    # Other parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    # GUI-specific arguments
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI (simulator only)")

    args = parser.parse_args()

    if args.no_gui:
        # Run simulator without GUI
        sim_args = sys.argv[1:]
        if "--no-gui" in sim_args:
            sim_args.remove("--no-gui")
        cmd = [sys.executable, "decentralized_fl_sim.py"] + sim_args
        subprocess.run(cmd)
    else:
        print("=" * 60)
        print("FEDERATED LEARNING MONITOR")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Nodes: {args.num_nodes}")
        print(f"  Rounds: {args.rounds}")
        print(f"  Algorithm: {args.agg.upper()}")
        print(f"  Graph: {args.graph}")
        if args.attack_percentage > 0:
            print(f"  Attack: {args.attack_type} ({args.attack_percentage*100:.1f}%)")
        print("=" * 60)

        # Prepare simulator arguments
        sim_args = []
        for arg, value in vars(args).items():
            if arg == "no_gui":
                continue
            arg_name = "--" + arg.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    sim_args.append(arg_name)
            elif value is not None:
                sim_args.extend([arg_name, str(value)])

        # Start GUI monitor process
        print("\n[1/2] Starting GUI monitor...")
        gui_process = Process(target=run_gui)
        gui_process.start()

        # Wait for GUI to initialize and start listening
        print("Waiting for GUI to initialize...")
        gui_ready = False
        max_wait = 10  # Maximum 10 seconds
        for i in range(max_wait):
            try:
                # Try to connect to the GUI to see if it's ready
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(0.5)
                result = test_socket.connect_ex(('localhost', 5555))
                test_socket.close()

                if result == 0:  # Connection successful
                    gui_ready = True
                    print("GUI is ready!")
                    break
                else:
                    # Try other ports
                    for port in [5556, 5557, 5558, 5559]:
                        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        test_socket.settimeout(0.5)
                        result = test_socket.connect_ex(('localhost', port))
                        test_socket.close()
                        if result == 0:
                            print(f"GUI ready on port {port}!")
                            gui_ready = True
                            break
                    if gui_ready:
                        break
            except Exception:
                pass

            time.sleep(1)
            print(f"Waiting... ({i+1}/{max_wait})")

        if not gui_ready:
            print("Warning: GUI may not be fully ready, but proceeding...")

        # Start simulator
        print("[2/2] Starting simulator with GUI integration...")
        sim_process = Process(target=run_simulator, args=(sim_args,))
        sim_process.start()

        # Handle termination
        def signal_handler(sig, frame):
            print("\nShutting down...")
            sim_process.terminate()
            gui_process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Wait for processes
            sim_process.join()
            print("\nSimulation completed. Close GUI window to exit.")
            gui_process.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            sim_process.terminate()
            gui_process.terminate()


if __name__ == "__main__":
    main()