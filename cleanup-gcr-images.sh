#!/bin/bash
# Usage: ./cleanup-gcr-images.sh <IMAGE_NAME> <KEEP_COUNT>
# Example: ./cleanup-gcr-images.sh gcr.io/tikr-mvp/tikr-mvp 2

IMAGE=$1
KEEP=$2

if [[ -z "$IMAGE" || -z "$KEEP" ]]; then
  echo "Usage: $0 <IMAGE_NAME> <KEEP_COUNT>"
  exit 1
fi

echo "Listing all tags for $IMAGE ..."
TAGS=$(gcloud container images list-tags $IMAGE \
    --sort-by=TIMESTAMP \
    --format='get(TAGS)' \
    | grep -v '^$' )

# Convert tags to array
TAGS_ARRAY=($TAGS)

# Count tags
TOTAL=${#TAGS_ARRAY[@]}

echo "Found $TOTAL tagged images."

# Calculate how many to delete
TO_DELETE=$((TOTAL - KEEP))

if [[ $TO_DELETE -le 0 ]]; then
  echo "Nothing to delete. Keeping all $TOTAL tags."
  exit 0
fi

echo "Deleting $TO_DELETE old tag(s)..."

# Loop through oldest tags first
for ((i=0; i<TO_DELETE; i++)); do
  TAG=${TAGS_ARRAY[$i]}
  echo "Deleting tag: $TAG"
  gcloud container images delete $IMAGE:$TAG --quiet
done

echo "Cleanup complete. Kept latest $KEEP tags."
