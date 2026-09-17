def decide_action(evidence):

    if evidence["sufficient"]:
        return {
            "decision": "auto_handle",
            "reason": evidence["reason"]
        }

    return {
        "decision": "escalate",
        "reason": evidence["reason"]
    }


if __name__ == "__main__":

    from retriever import retrieve_similar_cases
    from evidence_checker import check_evidence
    from response_generator import generate_response

    test_messages = [
        "My Spotify app keeps crashing",
        "What is Spotify Wrapped?",
        "I was charged twice for my subscription"
    ]

    for message in test_messages:

        results, _ = retrieve_similar_cases(
            message,
            top_k=5
        )

        evidence = check_evidence(
            message,
            results
        )

        decision = decide_action(evidence)

        print("\n" + "=" * 60)
        print("CUSTOMER:", message)
        print("EVIDENCE:", evidence)
        print("DECISION:", decision)

        if decision["decision"] == "auto_handle":

            response = generate_response(
                message,
                results
            )

            print("GENERATED RESPONSE:", response)

        else:

            print("ESCALATION: Human review required.")