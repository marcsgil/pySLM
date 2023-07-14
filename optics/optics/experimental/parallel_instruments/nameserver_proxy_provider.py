from __future__ import annotations

from multiprocessing import get_context, Pipe
import logging
import Pyro5.api
import Pyro5.core
import Pyro5.client
import Pyro5.errors
import Pyro5.nameserver
Pyro5.config.COMPRESSION = False
Pyro5.config.SERIALIZER = 'msgpack'

log = logging.getLogger(__name__)


__all__ = ['NameServerProxyProvider']


class NameServerProxyProvider:
    def __init__(self):
        self.__uri = None
        self.__proxy = None
        self.__parent_conn = self.__child_conn = None
        self.__process = None

    @property
    def proxy(self) -> Pyro5.client.Proxy:
        return self.__proxy

    def start(self) -> NameServerProxyProvider:
        """Starts a new nameserver as a separate process."""
        try:
            self.__proxy = Pyro5.core.locate_ns()
            log.info(f'Located nameserver, created proxy {self.__proxy}')
        except Pyro5.errors.NamingError:
            log.info('Creating nameserver...')
            self.__parent_conn, self.__child_conn = Pipe()
            ctx = get_context('spawn')
            self.__process = ctx.Process(target=self._run_nameserver, args=(self.__parent_conn,))
            log.info('Starting new nameserver process...')
            self.__process.start()
            self.__uri = self.__child_conn.recv()  # Not exposed publicly.
            log.info(f'Nameserver has uri = {self.__uri}')
            self.__proxy = Pyro5.client.Proxy(self.__uri)
        return self

    def stop(self):
        if self.__child_conn is not None:
            self.__child_conn.send('EXIT')
            self.__process.join()

    def __enter__(self) -> NameServerProxyProvider:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def _run_nameserver(parent_conn):
        """The code to be run as a separate process."""
        ns_uri, ns_daemon, bcserver = Pyro5.nameserver.start_ns()  # prepares the nameserver
        parent_conn.send(ns_uri)
        ns_daemon.requestLoop(lambda: not parent_conn.poll())
        log.info('Exited ns_deamon.requestloop')


if __name__ == '__main__':
    log.info('Starting nameserver...')
    with NameServerProxyProvider() as nameserver:
        log.info(f'Nameserver {nameserver} running with proxy {nameserver.proxy}.')
        input('Hit ENTER to exit.')
