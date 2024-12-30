import multiprocessing
import asyncio
import os
import zmq.asyncio

# Importing components
from analysis.strategy.optimizer.broker import broker
from analysis.strategy.optimizer.worker import worker, workers
from analysis.strategy.optimizer.collector import oracle
from data.ohlcv_repository import ohlcv_repository
from util import get_logger

# Configure root logger
logger = get_logger('main', level="INFO")

# Define task list for the broker
CHUNK_LENGTH = 1000
TASK_LIST = [f"Task-{i}" for i in range(10)]

# BROKER_ADDRESS = "ipc:///tmp/broker.ipc"
# ORACLE_ADDRESS = "ipc:///tmp/oracle.ipc"

BROKER_ADDRESS = "tcp://localhost:5555"
ORACLE_ADDRESS = "tcp://localhost:5556"
OHLCV_REPOSITORY_ADDRESS = "tcp://localhost:5557"


def run_broker(ctx: zmq.asyncio.Context | None = None) -> None:
    """Run the broker."""
    try:
        context = ctx or zmq.asyncio.Context()
        asyncio.run(
            broker(
                ctx=context,
                task_list=TASK_LIST,
                addr=BROKER_ADDRESS
                )
            )
    except KeyboardInterrupt:
        logger.info("[MAIN] Broker interrupted. Shutting down...")


def run_oracle(ctx: zmq.asyncio.Context | None = None) -> None:
    """Run the Oracle (Sink)."""
    try:
        context = ctx or zmq.asyncio.Context()
        asyncio.run(oracle(context, ORACLE_ADDRESS))
    except KeyboardInterrupt:
        logger.info("[MAIN] Oracle interrupted. Shutting down...")


def run_ohlcv_repository(ctx: zmq.asyncio.Context | None = None) -> None:
    """Run the OHLCV repositorys."""
    try:
        context = ctx or zmq.asyncio.Context()
        asyncio.run(ohlcv_repository(context, OHLCV_REPOSITORY_ADDRESS))
    except KeyboardInterrupt:
        logger.info("[MAIN] OHLCV repository interrupted. Shutting down...")


def run_main_components():
    """Run all main components inone process."""
    # context = zmq.asyncio.Context()
    ...  # Add more components here...


def run_worker(worker_id):
    """Run a worker in a separate process."""
    try:
        context = zmq.asyncio.Context()
        asyncio.run(
            worker(
                ctx=context,
                worker_id=worker_id,
                broker_address=BROKER_ADDRESS,
                oracle_address=ORACLE_ADDRESS,
                ohlcv_repository_address=OHLCV_REPOSITORY_ADDRESS,
                )
            )
    except KeyboardInterrupt:
        logger.info(f"[MAIN] Worker {worker_id} interrupted. Shutting down...")


def run_a_bunch_of_workers(base_id: str, num_workers: int):
    """Run multiple workers in one process."""
    worker_ids = [f"{base_id}-{i+1}" for i in range(num_workers)]

    try:
        context = zmq.asyncio.Context()
        asyncio.run(
            workers(
                context,
                worker_ids,
                BROKER_ADDRESS,
                ORACLE_ADDRESS,
                OHLCV_REPOSITORY_ADDRESS,
                len(worker_ids),
                )
            )
    except KeyboardInterrupt:
        logger.info("[MAIN] Some workers interrupted. Shutting down...")


def main():
    """Main entry point for the system."""
    try:
        logger.info("[MAIN] Starting the system...")

        # Determine number of workers based on CPU cores
        # Default to 4 if os.cpu_count() returns None
        num_workers = os.cpu_count() or 4
        logger.info(f"[MAIN] Number of worker processes: {num_workers}")

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

        # Start OHLCV Repository Process
        ohlcv_repository_process = multiprocessing.Process(
            target=run_ohlcv_repository,
            name="OHLCV Repository"
        )
        processes.append(ohlcv_repository_process)

        # Start Worker Processes
        for i in range(num_workers):
            worker_process = multiprocessing.Process(
                target=run_a_bunch_of_workers,
                name=f"Team-{i+1}",
                args=(str(i), 1,)
            )
            processes.append(worker_process)

        # Start all processes
        for process in processes:
            process.start()
            logger.debug(f"[MAIN] Started process: {process.name} (PID: {process.pid})")

        # Monitor processes
        for process in processes:
            process.join()
            logger.debug(f"[MAIN] Process {process.name} has finished.")

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
