import json
import datetime
import uuid

class PipelineLogger:
    def __init__(self, log_file_path="pipeline_log.jsonl"):
        self.log_file_path = log_file_path

    def _log_event(self, event_type: str, task_id: str, payload: dict):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "event_id": str(uuid.uuid4()),
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload
        }
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_task_start(self, task_id: str, problem_description: str, true_answer: any = None, initial_state: any = None):
        """Logs the beginning of a new task or problem-solving instance."""
        payload = {
            "problem_description": problem_description,
            "true_answer": true_answer,
            "initial_state": initial_state
        }
        self._log_event("task_start", task_id, payload)

    def log_stage_processing(self, task_id: str, stage_name: str, input_data: any, generated_answer: any, 
                             is_correct: bool, comparison_details: dict = None, execution_details: dict = None):
        """Logs the results of a specific processing stage."""
        payload = {
            "stage_name": stage_name,
            "input_data": input_data,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "comparison_details": comparison_details if comparison_details else {},
            "execution_details": execution_details if execution_details else {}
        }
        self._log_event("stage_processing", task_id, payload)

    def log_reflection_step(self, task_id: str, stage_name_before_reflection: str,
                            previous_generated_answer: any, previous_answer_correctness: bool,
                            reflection_prompt_or_trigger: any, reflection_content: str,
                            new_generated_answer: any, new_answer_correctness: bool,
                            impact_assessment: str, # e.g., "positive", "negative", "neutral"
                            transition: str # e.g., "wrong -> right", "correct -> correct_improved", "wrong -> wrong"
                            ):
        """Logs a reflection step and its impact."""
        payload = {
            "stage_name_before_reflection": stage_name_before_reflection,
            "previous_generated_answer": previous_generated_answer,
            "previous_answer_correctness": previous_answer_correctness,
            "reflection_prompt_or_trigger": reflection_prompt_or_trigger,
            "reflection_content": reflection_content,
            "new_generated_answer": new_generated_answer,
            "new_answer_correctness": new_answer_correctness,
            "impact_assessment": impact_assessment,
            "transition": transition
        }
        self._log_event("reflection_step", task_id, payload)

    def log_error_classification(self, task_id: str, generated_answer_reference: any, 
                                 classification_method: str, # "heuristic", "llm_as_judge", "task_specific_validation"
                                 method_details: dict, # e.g., {"heuristic_rules": ["missing_numbers"], "llm_prompt": "...", "llm_response": "..."}
                                 error_type: str, 
                                 error_subtype: str = None,
                                 is_truly_error: bool = None # Optional: Ground truth if available for this classification step
                                 ):
        """Logs the classification of an error."""
        payload = {
            "generated_answer_reference": generated_answer_reference,
            "classification_method": classification_method,
            "method_details": method_details,
            "error_type": error_type,
            "error_subtype": error_subtype,
        }
        if is_truly_error is not None:
            payload["is_truly_error_according_to_ground_truth"] = is_truly_error
            
        self._log_event("error_classification", task_id, payload)

    def log_analysis_summary(self, task_id: str, summary_stats: dict, notes: str = None):
        """Logs summary statistics or insights from the analysis module for a specific task or run."""
        payload = {
            "summary_stats": summary_stats,
            "notes": notes
        }
        self._log_event("analysis_summary", task_id, payload)
        
    def log_task_end(self, task_id: str, final_answer: any, final_correctness: bool, 
                     total_stages: int = None, total_reflections: int = None, 
                     overall_status: str = "completed"):
        """Logs the end of a task, including its final outcome and summary metrics."""
        payload = {
            "final_answer": final_answer,
            "final_correctness": final_correctness,
            "total_stages": total_stages,
            "total_reflections": total_reflections,
            "overall_status": overall_status # e.g., "completed", "failed", "aborted"
        }
        self._log_event("task_end", task_id, payload)

    def log_generic_event(self, task_id: str, event_name: str, details: dict):
        """Logs a generic event or piece of information not covered by other methods."""
        self._log_event(event_name, task_id, details)

if __name__ == '__main__':
    # Example Usage:
    logger = PipelineLogger(log_file_path="example_pipeline_log.jsonl")
    
    current_task_id = "task_12345"
    
    # 1. Log task start
    logger.log_task_start(
        task_id=current_task_id,
        problem_description="Solve the equation 2x + 5 = 15.",
        true_answer="x = 5",
        initial_state={"equation": "2x + 5 = 15"}
    )
    
    # 2. Log a processing stage
    logger.log_stage_processing(
        task_id=current_task_id,
        stage_name="InitialAttempt",
        input_data={"equation": "2x + 5 = 15"},
        generated_answer="x = 10",
        is_correct=False,
        comparison_details={"method": "direct_comparison", "difference": "value_mismatch"},
        execution_details={"model_used": "model_v1", "time_taken_ms": 150}
    )
    
    # 3. Log a reflection step
    logger.log_reflection_step(
        task_id=current_task_id,
        stage_name_before_reflection="InitialAttempt",
        previous_generated_answer="x = 10",
        previous_answer_correctness=False,
        reflection_prompt_or_trigger="The answer was incorrect. Re-check the arithmetic.",
        reflection_content="The error was in subtracting 5. 15 - 5 = 10. Then 10 / 2 = 5.",
        new_generated_answer="x = 5",
        new_answer_correctness=True,
        impact_assessment="positive",
        transition="wrong -> right"
    )
    
    # 4. Log error classification (if the initial answer was processed by classification)
    # This would typically be for an incorrect answer before reflection, or if reflection still leads to an error.
    logger.log_error_classification(
        task_id=current_task_id,
        generated_answer_reference="x = 10 (from InitialAttempt)",
        classification_method="llm_as_judge",
        method_details={
            "llm_prompt": "Is 'x=10' a correct solution for '2x+5=15'? If not, what type of error is it?",
            "llm_response": "No, it's a math error. Specifically, an arithmetic error in subtraction or division."
        },
        error_type="Math error",
        error_subtype="Arithmetic error"
    )
    
    # 5. Log another stage (e.g., if reflection was one stage, and now there's a final check)
    logger.log_stage_processing(
        task_id=current_task_id,
        stage_name="PostReflectionCheck",
        input_data={"equation": "2x + 5 = 15", "reflected_solution": "x = 5"},
        generated_answer="x = 5",
        is_correct=True,
        comparison_details={"method": "direct_comparison", "difference": "none"}
    )

    # 6. Log analysis summary (optional, can be logged at various points or at the end)
    logger.log_analysis_summary(
        task_id=current_task_id,
        summary_stats={"accuracy_after_reflection": 1.0, "errors_corrected": 1},
        notes="Reflection successfully corrected the initial arithmetic error."
    )
    
    # 7. Log task end
    logger.log_task_end(
        task_id=current_task_id,
        final_answer="x = 5",
        final_correctness=True,
        total_stages=2, # InitialAttempt, PostReflectionCheck
        total_reflections=1,
        overall_status="completed"
    )

    print(f"Example logs written to example_pipeline_log.jsonl") 