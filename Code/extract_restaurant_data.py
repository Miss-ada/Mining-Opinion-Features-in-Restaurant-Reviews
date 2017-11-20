import json, os
import pandas as pd

def load_json(file_name):
    json_list = []
    for line in open(file_name, 'r'):
        json_list.append(json.loads(line))
    return json_list


def get_business_id(parsed_json):
    return parsed_json['business_id']


def get_business_name(parsed_json):
    return parsed_json['name']


def get_business_category(business_json):
    if 'categories' in business_json:
        return business_json['categories']


def get_stars(review_json):
    if 'stars' in review_json:
        return review_json['stars']


def get_review(review_json):
    if 'text' in review_json:
        return review_json['text']


def get_all_restaurant_id_name(restaurant_json):
    restaurant_dict = {}
    for restaurant in restaurant_json:
        restaurant_id = get_business_id(restaurant)
        if restaurant_id not in restaurant_dict:
            restaurant_dict[restaurant_id] = get_business_name(restaurant)
    return restaurant_dict


def write_json(json_list, output_file):
    # with open(output_file, 'w') as outfile:
    #     outfile.write(json.dumps(json_list))
    with open(output_file, 'w') as outfile:
        strs = [json.dumps(dic) for dic in json_list]
        # s = "[%s]" % ",\n".join(strs)
        s = '\n'.join(strs)
        outfile.write(s)



if __name__ == '__main__':
    business_file = './business.json'
    review_file = './review.json'
    output_restaurant_json = 'restaurant.json'
    output_restaurant_review = 'restaurant_review.csv'

    # Extract all restaurants info
    if not os.path.isfile(output_restaurant_json):
        restaurant_json = []
        business_json = load_json(business_file)
        for business in business_json:
            if 'Restaurants' in get_business_category(business):
                restaurant_json.append(business)
        print('restaurant data length: ', len(restaurant_json))

        # write to a file
        write_json(restaurant_json, output_restaurant_json)

    # Extract all restaurant reviews according to id
    else:
        # Load file if already exists
        restaurant_json = load_json(output_restaurant_json)
        review_json = load_json(review_file)
        restaurant_dict = get_all_restaurant_id_name(restaurant_json)
        print('length of id list: ', len(restaurant_dict))

        restaurant_review = []
        for review in review_json:
            review_id = get_business_id(review)
            if review_id in restaurant_dict:
                restaurant_review.append((review_id, restaurant_dict[review_id], get_stars(review), get_review(review)))
        print('length of reviews: ', len(restaurant_review))
        # print(restaurant_review)
        # write to a csv
        df = pd.DataFrame(restaurant_review)
        df.to_csv(output_restaurant_review, header=['restaurant_id', 'name', 'stars', 'review'], index=False)
