import unittest
from unittest.mock import MagicMock, patch
from firewall.agent_define import AgentGuard
from Agent_Firewall.database.models import RequestLog, ContentLog

class TestAgentGuard(unittest.TestCase):
    @patch('firewall.agent_define.completion')
    @patch('firewall.agent_define.cost_calculator.completion_cost')
    def test_chat_logging(self, mock_cost, mock_completion):
        # 1. Setup Mocks
        # Mock LLM Response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I am a mock response."
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_completion.return_value = mock_response

        # Mock Cost
        mock_cost.return_value = 0.0005

        # Mock Database Session
        mock_db = MagicMock()

        # 2. Execute
        guard = AgentGuard()
        messages = [{"role": "user", "content": "Hello"}]
        guard.chat(user_id="test_user", messages=messages, db=mock_db)

        # 3. Verify
        # Check LLM was called
        mock_completion.assert_called_once()
        
        # Check DB was used to add logs
        # We expect 2 adds: RequestLog and ContentLog
        self.assertEqual(mock_db.add.call_count, 2)
        
        # Verify RequestLog content
        # args[0][0] is the first argument of the first call, etc.
        # We need to find which call was for RequestLog
        added_objects = [call[0][0] for call in mock_db.add.call_args_list]
        req_log = next((obj for obj in added_objects if isinstance(obj, RequestLog)), None)
        content_log = next((obj for obj in added_objects if isinstance(obj, ContentLog)), None)

        self.assertIsNotNone(req_log)
        self.assertIsNotNone(content_log)
        
        self.assertEqual(req_log.user_id, "test_user")
        self.assertEqual(req_log.total_cost, 0.0005)
        self.assertEqual(content_log.user_query, "Hello")
        self.assertEqual(content_log.bot_response, "I am a mock response.")
        
        # Check commit was called
        mock_db.commit.assert_called_once()
        print("\n✅ Test Passed: Logic flow validates successfully.")

if __name__ == '__main__':
    unittest.main()
