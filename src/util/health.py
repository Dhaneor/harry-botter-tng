#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:12:20 2022

@author dhaneor
"""

import os
import random
from multiprocessing import Process
import psutil
import time
from threading import Thread  # noqa: F401


class Health:  # Thread
    def __init__(self):
        self.proc_cpu_times = {}
        self.last_call = 0

    def run(self):
        pass

    def get_stats(self, pid: int):
        if not pid:
            return {
                "cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
                "cpu_freq": psutil.cpu_freq(),
                "network_io": psutil.net_io_counters(),
                "network_connections": self.get_network_connections(),
                "python_processes": self.get_all_processes_info(),
            }
        else:
            return {
                "cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
                "cpu_freq": psutil.cpu_freq(),
                "network_io": psutil.net_io_counters(),
                "network_connections": self.get_network_connections(),
                "current_process": self.get_info_for_current_process(),
            }

    def get_network_connections(self):
        try:
            return psutil.net_connections()
        except psutil.AccessDenied:
            return "access denied"

    def get_info_for_current_process(self):
        current_pid = os.getpid()
        # process = psutil.Process(current_pid)
        # return self._get_process_info(process, current_pid)

        for process in psutil.process_iter():
            with process.oneshot():
                try:
                    pid = process.pid
                except psutil.AccessDenied:
                    pid = None

                if pid == current_pid:
                    return self._get_process_info(process, pid)

    def get_all_processes_info(self):
        processes = []

        for process in psutil.process_iter():
            with process.oneshot():
                try:
                    pid = process.pid
                except psutil.AccessDenied:
                    pid = None

                if info := self._get_process_info(process, pid):
                    processes.append(info)

        return [p for p in processes if "python" in p["name"]]

    def _get_process_info(self, process: psutil.Process, pid):
        with process.oneshot():
            try:
                name = process.name()
            except psutil.AccessDenied:
                name = "access denied"

            try:
                cpu_usage = process.cpu_percent()
            except psutil.AccessDenied:
                cpu_usage = "access denied"

            total_time_passed = time.time() - self.last_call
            cpu_times = {"system": 0, "user": 0}
            try:
                cpu_times = process.cpu_times()
            except Exception:
                pass
            finally:
                last_call = time.time()

            cpu_percent = -1
            if (current := cpu_times) and (last := self.proc_cpu_times.get(pid)):
                try:
                    spent_time = sum(
                        [
                            current.system - last.system,
                            current.user - last.user,
                            current.children_system - last.children_system,
                            current.children_user - last.children_user,
                        ]
                    )
                    if total_time_passed > 0:
                        cpu_percent = round(100 * (spent_time / total_time_passed), 4)
                except Exception:
                    pass

            self.proc_cpu_times[pid] = cpu_times
            self.last_call = last_call

            try:
                memory_usage = process.memory_full_info().uss
            except psutil.AccessDenied:
                memory_usage = 0

            return {
                "name": name,
                "cpu_usage": cpu_usage,
                "cpu_percent": cpu_percent,
                "memory_usage": memory_usage,
            }


def heavy_work(n):
    def the_work(n):
        for x in range(n):
            _ = x**2 + x**0.5 + x**0.3 - x**0.8
            time.sleep(random.random())

    for _ in range(20):
        the_work(n)
        time.sleep(random.randint(1, 5))


if __name__ == "__main__":
    h = Health()

    threads = []
    for _ in range(4):
        t = Process(target=heavy_work, args=[100_000_000], daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            pid = os.getpid()
            st = time.time()
            stats = h.get_stats(pid)
            process = stats["current_process"]
            et = time.time()
            if process:
                # cpu_usage = process['cpu_usage']
                # cpu = stats['cpu_percent']
                # memory_usage = process['memory_usage']
                # os.system('clear')
                # print(
                #     f'cpu_usage of current process ({pid}): {cpu_usage} - {cpu} '\
                #     f'with memory usage: {memory_usage} ({round((et-st)*1000, 2)})'
                # )
                print(process)
                time.sleep(1)
            else:
                os.system("clear")
                print("unknown")
    except KeyboardInterrupt:
        [t.join() for t in threads]
        time.sleep(6)
