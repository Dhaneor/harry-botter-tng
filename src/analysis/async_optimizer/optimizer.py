import multiprocessing
import asyncio
import os
import zmq.asyncio

# Importing components
from analysis.async_optimizer.broker import broker
from analysis.async_optimizer.worker import worker
from analysis.async_optimizer.collector import oracle
from util import get_logger

# Configure root logger
logger = get_logger('main', level="INFO")

# Define task list for the broker
TASK_LIST = [f"Task-{i}" for i in range(1, 21)]  # Example task list with 20 tasks


def run_broker():
    """Run the broker in a separate process."""
    try:
        asyncio.run(broker(TASK_LIST))
    except KeyboardInterrupt:
        logger.info("[MAIN] Broker interrupted. Shutting down...")


def run_oracle():
    """Run the Oracle (Sink) in a separate process."""
    try:
        context = zmq.asyncio.Context()
        asyncio.run(oracle(context))
    except KeyboardInterrupt:
        logger.info("[MAIN] Oracle interrupted. Shutting down...")


def run_worker(worker_id):
    """Run a worker in a separate process."""
    try:
        context = zmq.asyncio.Context()
        asyncio.run(worker(context, worker_id=worker_id))
    except KeyboardInterrupt:
        logger.info(f"[MAIN] Worker {worker_id} interrupted. Shutting down...")


def main():
    """Main entry point for the system."""
    try:
        logger.info("[MAIN] Starting the system...")

        # Determine number of workers based on CPU cores
        num_workers = os.cpu_count() or 4  # Default to 4 if os.cpu_count() returns None
        logger.info(f"[MAIN] Number of workers: {num_workers}")

        # Create processes
        processes = []

        # Start Broker Process
        broker_process = multiprocessing.Process(
            target=run_broker,
            name="Broker"
        )
        processes.append(broker_process)

        # Start Oracle Process
        oracle_process = multiprocessing.Process(
            target=run_oracle,
            name="Oracle"
        )
        processes.append(oracle_process)

        # Start Worker Processes
        for i in range(num_workers):
            worker_process = multiprocessing.Process(
                target=run_worker,
                name=f"Worker-{i+1}",
                args=(f"worker-{i+1}",)
            )
            processes.append(worker_process)

        # Start all processes
        for process in processes:
            process.start()
            logger.info(f"[MAIN] Started process: {process.name} (PID: {process.pid})")

        # Monitor processes
        for process in processes:
            process.join()
            logger.info(f"[MAIN] Process {process.name} has finished.")

    except KeyboardInterrupt:
        logger.info("[MAIN] Interrupted by user. Shutting down all processes...")
        for process in processes:
            process.terminate()
            logger.info(f"[MAIN] Terminated process: {process.name}")
    except Exception as e:
        logger.error(f"[MAIN] An error occurred: {e}")
    finally:
        logger.info("[MAIN] System shutdown complete.")


if __name__ == "__main__":
    main()
