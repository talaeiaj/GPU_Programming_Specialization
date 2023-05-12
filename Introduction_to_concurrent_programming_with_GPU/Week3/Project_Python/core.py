# Based on RealPython Threading Example page at https://realpython.com/intro-to-python-threading/
import logging
import random
import time
import argparse
import pydash as _


def thread_function(index):
    logging.info("Thread %d: starting", index)
    time.sleep(1)
    logging.info("Thread %d: finishing", index)


def critical_section_acquire_release(name, sync_object):
    # Add random amount of time for sleep so that execution may be random
    time.sleep(random.randint(0, 10))
    sync_object.acquire()
    logging.info("critical_section_acquire_release thread: %d acquired synchronization object.", name)
    thread_function(name)
    sync_object.release()
    logging.info("critical_section_acquire_release thread: %d released synchronization object.", name)


class Core:

    file_username = None
    username_arg = None
    num_threads = 1
    part_id = None

    def __init__(self, args_list=None):
        self.parser = argparse.ArgumentParser(description='Process command-line arguments')
        for arg in args_list:
            self.add_arg_parser_argument(arg)
        self.read_user_file()

    def read_user_file(self):
        file = open(".user", "r")
        self.file_username = file.readline()

    def test_username_equality(self, const_username):
        return self.file_username == self.username_arg and self.file_username == const_username

    def parse_args(self, args):
        namespace = self.parser.parse_args(args=args)
        if namespace:
            self.num_threads = int(_.get(namespace, 'num_threads', 1))
            self.username_arg = str(_.get(namespace, 'user', None))
            self.part_id = str(_.get(namespace, 'part_id', None))

    def add_arg_parser_argument(self, arg):
        self.parser.add_argument(arg[0], dest=arg[1], default=arg[2], help=arg[3])
