from gfootball.env import football_action_set
from gfootball.env import player_base

class Player(player_base.PlayerBase):
  """Lazy player not moving at all."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)

  def take_action(self, observations):
    return [football_action_set.action_idle] * len(observations)
