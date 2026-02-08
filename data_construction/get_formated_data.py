import os
import json
import argparse

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    entities = ["award", "region", "language", "office", "river"]

    def format_fact(entity, fact):
        """
        Format according to different entity types.
        """
        # Only simple formatting here, can be extended as needed
        if entity == "award":
            # Only return {year: [names, ...], ...}
            years = fact.get("years", {})
            year_to_names = {}
            for year, winners in years.items():
                # winners is a list, each element is [name, qid, type, country, country_qid]
                names = list(set([winner[0] for winner in winners if winner and len(winner) > 0]))
                year_to_names[year] = names
            return year_to_names
        elif entity == "region":
            # Country format should be: {"CountryName": {"Region1": [low, high], ...}}
            # The input fact is all the info for a country, return in original format
            divisions = fact.get("divisions", [])
            name_area_map = {}
            for d in divisions:
                name = d.get("name")
                area = d.get("area")
                if name is not None and area is not None:
                    try:
                        area_int = int(area)
                        area_str = str(area_int)
                        digits = len(area_str)
                        if digits > 2:
                            prefix = int(area_str[:2])
                            lower = prefix * 10 ** (digits - 2)
                            upper = (prefix + 1) * 10 ** (digits - 2)
                        elif digits == 2:
                            prefix = int(area_str[0])
                            lower = prefix * 10
                            upper = (prefix + 1) * 10
                        else:
                            lower = area_int
                            upper = area_int + 1
                        name_area_map[name] = [lower, upper]
                    except Exception as e:
                        name_area_map[name] = [area, area]
            return name_area_map

        elif entity == "language":
            # Return a deduplicated list of country names
            countries = fact.get("countries", [])
            name_set = set()
            for c in countries:
                if isinstance(c, list) and len(c) > 0:
                    name_set.add(c[0])
                elif isinstance(c, str):
                    name_set.add(c)
            return list(name_set)
        elif entity == "office":
            years = fact.get("years", {})
            year_to_name = {}
            for year, holders in years.items():
                # holders is a list, each element is [name, qid, country, country_qid]
                if holders and len(holders) > 0 and len(holders[0]) > 0:
                    year_to_name[year] = holders[0][0]
            return year_to_name
        elif entity == "river":
            country_labels = set()
            # Extract 'label' from each element in countries
            for c in fact.get("countries", []):
                label = c.get("label")
                if label:
                    country_labels.add(label)
            return list(country_labels)
        else:
            return fact

    # input_dir and output_dir should be passed from arguments
    for entity in entities:
        input_file = os.path.join(input_dir, f"{entity}_structured_facts.json")
        output_file = os.path.join(output_dir, f"{entity}_structured_facts.json")
        if not os.path.exists(input_file):
            print(f"File does not exist: {input_file}")
            continue
        with open(input_file, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        # Format each entity id and write the whole dict out
        formatted_data = {}
        for k, v in data.items():
            formatted_data[k] = format_fact(entity, v)
        with open(output_file, "w", encoding="utf-8") as fout:
            json.dump(formatted_data, fout, ensure_ascii=False, indent=2)
        print(f"{entity} formatted and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format structured facts for entities.")
    # Use -- prefix to make it an optional argument
    parser.add_argument("--input_dir", type=str, help="Input directory input_dir", required=True)
    parser.add_argument("--output_dir", type=str, help="Output directory output_dir", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
