import re



def postprocess_generated_answer(prompt, answer, question, retrieved_context):
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    answer_lines = answer.split('\n')
    filtered_lines = [line for line in answer_lines if question.lower() not in line.lower()]
    context_lines = set([l.strip() for l in retrieved_context.split('\n') if l.strip()])
    filtered_lines = [line for line in filtered_lines if line.strip() not in context_lines]
    answer = '\n'.join(filtered_lines).strip()
    answer = re.sub(r"^A:\s*", "", answer)
    return answer






