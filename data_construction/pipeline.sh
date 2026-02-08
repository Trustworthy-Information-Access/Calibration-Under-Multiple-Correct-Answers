QA_COUNT=500
TOP_K=5000
VIEW_NUM=1000
ENTITY=all

POPULAR_DIR="./entity_popularity"
FACT_TOPK_VIEW_DIR="./entity_facts_topk${TOP_K}_view${VIEW_NUM}"
FACT_FILTER_DIR="./entity_facts_topk${TOP_K}_view${VIEW_NUM}_filter"
FACT_FILTER_FORMAT_DIR="./entity_facts_topk${TOP_K}_view${VIEW_NUM}_filter_format"
QA_DIR="./QA_topk${TOP_K}_view${VIEW_NUM}_num${QA_COUNT}"

# # # Fetch entity popularity (uncomment if you need to refresh)
python ./fetch_entity_popularity.py \
  --limit=20000 \
  --output_dir="$POPULAR_DIR" \
  --limit=20

# # # # Fetch structured facts and sub-entity mapping
python ./fetch_facts.py \
  --entity_popularity_dir "$POPULAR_DIR" \
  --output_dir "$FACT_TOPK_VIEW_DIR" \
  --top_k "$TOP_K" \
  --view_num "$VIEW_NUM" \
  --limit 1000 \
  --entity "$ENTITY" \
  --outer_workers 16 \
  --inner_workers 16 \

# # # # # Filter out invalid structured facts
python ./filter.py \
  --entity "$ENTITY" \
  --input_dir "$FACT_TOPK_VIEW_DIR" \
  --output_dir "$FACT_FILTER_DIR"

python ./get_formated_data.py \
  --input_dir "$FACT_FILTER_DIR" \
  --output_dir "$FACT_FILTER_FORMAT_DIR" \

# Human check...

python ./generate_qa.py \
  --min_answers 1 \
  --max_answers 1 \
  --qa_count $QA_COUNT \
  --input_dir "$FACT_FILTER_FORMAT_DIR" \
  --output_dir "$QA_DIR" \
  --entity "$ENTITY" 

python ./generate_qa.py \
  --min_answers 2 \
  --max_answers 2 \
  --qa_count $QA_COUNT \
  --input_dir "$FACT_FILTER_FORMAT_DIR" \
  --output_dir "$QA_DIR" \
  --entity "$ENTITY" 

python ./generate_qa.py \
  --min_answers 4 \
  --max_answers 4 \
  --qa_count $QA_COUNT \
  --input_dir "$FACT_FILTER_FORMAT_DIR" \
  --output_dir "$QA_DIR" \
  --entity "$ENTITY" 

python ./generate_qa.py \
  --min_answers 6 \
  --max_answers 6 \
  --qa_count $QA_COUNT \
  --input_dir "$FACT_FILTER_FORMAT_DIR" \
  --output_dir "$QA_DIR" \
  --entity "$ENTITY" 
