from datasets import load_dataset
from examples.foa.utils import grade_answer
from examples.foa.utils import extract_last_boxed_answer, extract_answer_from_output

# Load dataset
dataset = load_dataset("hendrycks/competition_math", split='test', trust_remote_code=True)

# Extract the numerical solution 
dataset = dataset.map(lambda x: {"solution_num": extract_last_boxed_answer(x["solution"])})

# Iterate through dataset and grade answers
# First 5 entries are skipped because they are used for examples in prompts
for data_entry in dataset.iloc[5:]:
    print("Question:", data_entry["problem"])
    print("Solution:", data_entry["solution"])

    ### Here we would generate answers using the model
    # output = model.generate()....)
    # Let's use a dummy answer instead
    output = "This is a dummy output. The final answer is \\boxed{2}."

    # This will extract the final boxed answer from the output, but in your format, the output will be extracted differently.
    predicted_answer = extract_answer_from_output(output)
    correct_answer = data_entry["solution_num"]

    print("Ground truth answer:", correct_answer)
    print("Model answer:", output)

    is_correct = grade_answer(predicted_answer, correct_answer)
    print("Is correct:", is_correct)
    print("--")
    break
