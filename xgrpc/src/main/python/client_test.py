import unittest

from . import client

class TestClientConnection(unittest.TestCase):

  def test_can_create_session(self):
    client.run() 

