#!/bin/bash

# A script to track changes, update UPDATES.md, and commit.

set -e

# Check if inside a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1;
then
    echo "Error: Not in a git repository."
    exit 1
fi

# Get the root directory of the git repository
GIT_ROOT=$(git rev-parse --show-toplevel)
UPDATES_FILE="$GIT_ROOT/UPDATES.md"

# Arguments
DIFF_COMMAND="git diff --staged"

while getopts ":r:" opt; do
  case ${opt} in
    r)
      DIFF_COMMAND="git diff ${OPTARG}..HEAD"
      ;;
    
    ?)
      echo "Invalid option: -${OPTARG}" >&2
      echo "Usage: $(basename "$0") [-r <commit-hash>]"
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      echo "Usage: $(basename "$0") [-r <commit-hash>]"
      exit 1
      ;;
  esac
done

# Stage all changes
git add -u

# Get the diff output, filtering for non-code changes
DIFF=$(eval "$DIFF_COMMAND" | grep -E '^(\+\+\+|---|\+|\-|@@)' | grep -v '^[+ ]*$' | grep -v '^-*$')

if [ -z "$DIFF" ]; then
    echo "No relevant changes found. Exiting."
    exit 0
fi

# Use a temporary file to store Gemini's output
TEMP_MD=$(mktemp)
CURRENT_DATE=$(date '+%Y-%m-%d')

# Call Gemini to summarize the changes
gemini <<EOF > "$TEMP_MD"
You are an expert technical writer and code reviewer.

### CONTEXT ###
The user has provided a git diff output. Your task is to analyze the changes for errors, breakpoints, and then generate a summary suitable for a project update log.

### PRIMARY TASK ###
1.  **Review for errors and breakpoints**:
    - Check for obvious errors in the code (e.g., syntax errors). If you find any, output ONLY a description of the error and the file it is in, prefixed with "ERROR: ".
    - Check for breakpoints (e.g., `pdb.set_trace()`, `breakpoint()`). If you find any, you will mention it later.

2.  **Summarize and categorize changes**:
    - If no errors are found, summarize and categorize the changes from the git diff.
    - Use a category for each change (e.g., 'What's New', 'Bugfix', 'Refactor').
    - If you found breakpoints, add a 'Warnings' section at the top of your summary, listing the files containing breakpoints.

### SPECIFICATIONS & INSTRUCTIONS ###
- Group changes by category, then by file.
- Do not include spaces, newlines, or whitespace-only changes.
- Do not include any conversational text outside the formatted summary.
- The most recent update must be placed at the top of the file.

### OUTPUT FORMAT & CONSTRAINTS ###
- If errors are found, output ONLY the error description (e.g., "ERROR: Syntax error in main.py").
- Otherwise, provide your response exclusively as the raw text of the summary.
- DO NOT include any explanations or introductory sentences.
- The current date is $CURRENT_DATE.

Act autonomously. Do not ask for clarification. Begin analysis immediately when invoked.

${DIFF}
EOF

# Check for errors reported by Gemini
if grep -q "^ERROR:" "$TEMP_MD"; then
    cat "$TEMP_MD"
    rm "$TEMP_MD"
    echo "Errors found by AI. Aborting update."
    exit 1
fi

SUMMARY=$(cat "$TEMP_MD")

# Prepend the new content to UPDATES.md
if [ -f "$UPDATES_FILE" ]; then
    CURRENT_CONTENT=$(cat "$UPDATES_FILE")
    echo -e "${SUMMARY}\n\n${CURRENT_CONTENT}" > "$UPDATES_FILE"
else
    echo -e "${SUMMARY}" > "$UPDATES_FILE"
fi

# Clean up temp file
rm "$TEMP_MD"

# Add the updated UPDATES.md to staged files
git add "$UPDATES_FILE"

# Generate commit message and commit
COMMIT_MSG=$(gemini <<EOF
### ROLE & PERSONA ###
You are an expert at generating git commit messages.

### CONTEXT ###
The user has provided a summary of code changes.

### PRIMARY TASK ###
Based on the summary, generate a concise and relevant git commit message.

### SPECIFICATIONS & INSTRUCTIONS ###
- The first line should be the subject, following Conventional Commits format (e.g., 'feat: add new feature').
- The subject should be a maximum of 50 characters.
- The body should contain a brief, 1-2 sentence description if necessary.

### EXAMPLE FORMAT ###
feat: add user authentication
This commit adds a new user authentication flow using OAuth 2.0.

### OUTPUT FORMAT & CONSTRAINTS ###
- Provide your response as the raw commit message.
- Do not include any explanations or conversational text.

Act autonomously. Do not ask for clarification. Begin analysis immediately when invoked.

${SUMMARY}
EOF
)

git commit -m "$COMMIT_MSG"

echo "Changes summarized, UPDATES.md updated, and committed."