"""
Data preprocessing utilities for 30ppl evaluation dataset.

This module provides functions to extract and preprocess various components
from the JSON files containing user comments, posts, and evaluation data.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class DataPreprocessor:
    """Main class for preprocessing evaluation dataset."""

    def __init__(self, data_dir: str = "data/30ppl_eval_v1_0728"):
        """
        Initialize the preprocessor.

        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = Path(data_dir)
        self.participants = self._get_participants()

    def _get_participants(self) -> List[str]:
        """Get list of participant directories."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")

        participants = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                participants.append(item.name)
        return sorted(participants)

    def load_participant_data(
        self, participant: str, split: str = "test"
    ) -> Dict[str, Any]:
        """
        Load JSON data for a specific participant.

        Args:
            participant: Participant username/directory name
            split: Data split ('test' or 'train')

        Returns:
            Dictionary containing the participant's data
        """
        file_path = self.data_dir / participant / f"{participant}-{split}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_user_metadata(
        self, participant: str, split: str = "test"
    ) -> Dict[str, Any]:
        """
        Extract user metadata from participant data.

        Args:
            participant: Participant username
            split: Data split to use

        Returns:
            Dictionary with user metadata
        """
        data = self.load_participant_data(participant, split)

        metadata = {
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "comment_count": data.get("comment_count"),
            "topic_count": data.get("topic_count"),
            "account_created_utc": data.get("account_created_utc"),
            "account_created_iso": data.get("account_created_iso"),
            "user_karma": data.get("user_karma"),
            "avg_comment_length": data.get("avg_comment_length"),
            "comment_frequency_monthly": data.get("comment_frequency_monthly"),
            "subreddits_commented": data.get("subreddits_commented"),
            "time_range": data.get("time_range"),
        }

        return metadata

    def extract_topics(
        self, participant: str, split: str = "test"
    ) -> List[Dict[str, Any]]:
        """
        Extract topic/post information from participant data.

        Args:
            participant: Participant username
            split: Data split to use

        Returns:
            List of topic dictionaries
        """
        data = self.load_participant_data(participant, split)
        topics = []

        for topic in data.get("topics", []):
            topic_info = {
                "subreddit": topic.get("subreddit"),
                "post_id": topic.get("post_id"),
                "post_title": topic.get("post_title"),
                "scenario_description": topic.get("scenario_description"),
                "post_created_utc": topic.get("post_created_utc"),
                "post_created_iso": topic.get("post_created_iso"),
                "post_flair": topic.get("post_flair"),
                "post_label": topic.get("post_label"),
                "comment_id": topic.get("comment_id"),
                "comment_text": topic.get("comment_text"),
                "comment_created_utc": topic.get("comment_created_utc"),
                "comment_time_iso": topic.get("comment_time_iso"),
                "comment_length": topic.get("comment_length"),
                "comment_score": topic.get("comment_score"),
                "comment_permalink": topic.get("comment_permalink"),
                "stance_label": topic.get("stance_label"),
            }
            topics.append(topic_info)

        return topics

    def extract_mcqs(
        self, participant: str, split: str = "test", question_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract MCQ questions and answers from participant data.

        Args:
            participant: Participant username
            split: Data split to use
            question_type: Filter by question type (e.g., 'stance', 'reasoning', 'emotion', 'claim')

        Returns:
            List of MCQ dictionaries
        """
        data = self.load_participant_data(participant, split)
        mcqs = []

        for topic in data.get("topics", []):
            post_id = topic.get("post_id")

            for mcq in topic.get("mcqs", []):
                mcq_info = {
                    "post_id": post_id,
                    "question_id": mcq.get("question_id"),
                    "question": mcq.get("question"),
                    "tag": mcq.get("tag"),
                    "options": mcq.get("options"),
                    "ground_truth": mcq.get("ground_truth"),
                    "explanation": mcq.get("explanation"),
                    "source": mcq.get("source"),
                    "question_type": mcq.get("question_type"),
                }

                # Filter by question type if specified
                if question_type is None or mcq_info["tag"] == question_type:
                    mcqs.append(mcq_info)

        return mcqs

    def extract_comment_analysis(
        self, participant: str, split: str = "test"
    ) -> List[Dict[str, Any]]:
        """
        Extract comment analysis data (stance, emotions, reasoning patterns).

        Args:
            participant: Participant username
            split: Data split to use

        Returns:
            List of comment analysis dictionaries
        """
        data = self.load_participant_data(participant, split)
        analysis = []

        for topic in data.get("topics", []):
            comment_data = {
                "post_id": topic.get("post_id"),
                "comment_id": topic.get("comment_id"),
                "comment_text": topic.get("comment_text"),
                "stance_label": topic.get("stance_label"),
                "post_label": topic.get("post_label"),
                "comment_length": topic.get("comment_length"),
                "comment_score": topic.get("comment_score"),
                "subreddit": topic.get("subreddit"),
                "mcq_count": len(topic.get("mcqs", [])),
                "mcq_tags": [mcq.get("tag") for mcq in topic.get("mcqs", [])],
            }
            analysis.append(comment_data)

        return analysis

    def get_split_info(self, participant: str) -> Dict[str, Any]:
        """
        Get train/test split information for a participant.

        Args:
            participant: Participant username

        Returns:
            Dictionary with split information
        """
        split_file = self.data_dir / participant / "split_log.json"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found")

        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_participant_summary(self, participant: str) -> Dict[str, Any]:
        """
        Create a comprehensive summary for a participant.

        Args:
            participant: Participant username

        Returns:
            Dictionary with participant summary
        """
        try:
            # Get metadata from test split
            metadata = self.extract_user_metadata(participant, "test")
            split_info = self.get_split_info(participant)

            # Get topics from both splits
            test_topics = self.extract_topics(participant, "test")
            train_topics = self.extract_topics(participant, "train")

            # Get MCQs from test split
            test_mcqs = self.extract_mcqs(participant, "test")

            summary = {
                "participant": participant,
                "metadata": metadata,
                "split_info": split_info,
                "topic_counts": {
                    "total": len(test_topics) + len(train_topics),
                    "test": len(test_topics),
                    "train": len(train_topics),
                },
                "mcq_count": len(test_mcqs),
                "mcq_types": list(set([mcq["tag"] for mcq in test_mcqs])),
                "subreddits": list(
                    set([topic["subreddit"] for topic in test_topics + train_topics])
                ),
                "stance_distribution": self._get_stance_distribution(
                    test_topics + train_topics
                ),
            }

            return summary

        except Exception as e:
            return {"participant": participant, "error": str(e)}

    def _get_stance_distribution(self, topics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate stance label distribution."""
        stances = [
            topic.get("stance_label") for topic in topics if topic.get("stance_label")
        ]
        distribution = {}
        for stance in stances:
            distribution[stance] = distribution.get(stance, 0) + 1
        return distribution

    def export_to_dataframe(self, component: str = "topics", split: str = "test"):
        """
        Export data component to pandas DataFrame for all participants.

        Args:
            component: Component to export ('topics', 'mcqs', 'metadata', 'analysis')
            split: Data split to use

        Returns:
            pandas DataFrame with the requested component data (if pandas available)
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for DataFrame export. Install with: pip install pandas"
            )

        all_data = []

        for participant in self.participants:
            try:
                if component == "topics":
                    data = self.extract_topics(participant, split)
                elif component == "mcqs":
                    data = self.extract_mcqs(participant, split)
                elif component == "metadata":
                    data = [self.extract_user_metadata(participant, split)]
                elif component == "analysis":
                    data = self.extract_comment_analysis(participant, split)
                else:
                    raise ValueError(f"Unknown component: {component}")

                # Add participant info to each record
                for record in data:
                    record["participant"] = participant
                    all_data.append(record)

            except Exception as e:
                print(f"Error processing {participant}: {e}")
                continue

        return pd.DataFrame(all_data)


def validate_data_integrity(
    data_dir: str = "data/30ppl_eval_v1_0728",
) -> Dict[str, Any]:
    """
    Validate data integrity across all participants.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with validation results
    """
    preprocessor = DataPreprocessor(data_dir)
    results = {
        "total_participants": len(preprocessor.participants),
        "valid_participants": 0,
        "invalid_participants": [],
        "file_issues": [],
        "data_issues": [],
    }

    for participant in preprocessor.participants:
        try:
            # Check if required files exist
            for split in ["train", "test"]:
                file_path = (
                    preprocessor.data_dir / participant / f"{participant}-{split}.json"
                )
                if not file_path.exists():
                    results["file_issues"].append(
                        f"Missing {split} file for {participant}"
                    )
                    continue

            # Check split log
            split_file = preprocessor.data_dir / participant / "split_log.json"
            if not split_file.exists():
                results["file_issues"].append(
                    f"Missing split_log.json for {participant}"
                )

            # Validate data structure
            test_data = preprocessor.load_participant_data(participant, "test")
            if "topics" not in test_data or not test_data["topics"]:
                results["data_issues"].append(f"No topics found for {participant}")

            results["valid_participants"] += 1

        except Exception as e:
            results["invalid_participants"].append(f"{participant}: {e}")

    return results


# Convenience functions for quick access
def quick_load_participant(
    participant: str,
    split: str = "test",
    data_dir: str = "data/30ppl_eval_v1_0728",
) -> Dict[str, Any]:
    """Quick function to load participant data."""
    preprocessor = DataPreprocessor(data_dir)
    return preprocessor.load_participant_data(participant, split)


def quick_extract_mcqs(
    participant: str,
    question_type: Optional[str] = None,
    data_dir: str = "data/30ppl_eval_v1_0728",
) -> List[Dict[str, Any]]:
    """Quick function to extract MCQs for a participant."""
    preprocessor = DataPreprocessor(data_dir)
    return preprocessor.extract_mcqs(participant, "test", question_type)


def get_all_participants(data_dir: str = "data/30ppl_eval_v1_0728") -> List[str]:
    """Get list of all participants in the dataset."""
    preprocessor = DataPreprocessor(data_dir)
    return preprocessor.participants


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(data_dir="data/30ppl_eval_v1_0728")

    # Print available participants
    print("Available participants:", len(preprocessor.participants))
    print("First 5:", preprocessor.participants[:5])

    # Example: Extract data for first participant
    if preprocessor.participants:
        participant = preprocessor.participants[0]
        print(f"\nExample data for {participant}:")

        # User metadata
        metadata = preprocessor.extract_user_metadata(participant)
        print(f"User karma: {metadata['user_karma']}")
        print(f"Comment count: {metadata['comment_count']}")

        # Topics
        topics = preprocessor.extract_topics(participant)
        print(f"Number of topics: {len(topics)}")

        # MCQs
        mcqs = preprocessor.extract_mcqs(participant)
        print(f"Number of MCQs: {len(mcqs)}")

        # MCQs by type
        stance_mcqs = preprocessor.extract_mcqs(participant, question_type="stance")
        print(f"Stance MCQs: {len(stance_mcqs)}")
