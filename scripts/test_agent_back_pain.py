import json

from agent.orchestrator import TriageAgentOrchestrator


def test_agent():
    orchestrator = TriageAgentOrchestrator()

    print("--- Test 1 : Plainte courte ---")
    input1 = "douleurs dorsales"
    result1 = orchestrator.run(input1)
    print(f"Input: '{input1}'")
    print(f"Result: {json.dumps(result1, indent=2)}")

    print("\n--- Test 2 : Plainte détaillée ---")
    input2 = "douleurs dorsales persistantes depuis deux semaines avec irradiations"
    result2 = orchestrator.run(input2)
    print(f"Input: '{input2}'")
    print(f"Result: {json.dumps(result2, indent=2)}")


if __name__ == "__main__":
    test_agent()
