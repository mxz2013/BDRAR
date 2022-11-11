import os
import cv2
import pandas as pd


def main():
    csv_name = 'select_best_shadow.csv'
    data_dir = '/Users/sky/Documents/data_glovo/select_shadow_after_inference2022_11_10_09_39_27'
    csv_path = os.path.join(data_dir, csv_name)
    shadow_mask_path = os.path.join(data_dir, 'with_shadow')
    img_names = os.listdir(shadow_mask_path)
    img_names.sort()

    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['img_name', 'best_shadow'])
        current_id = 0
    else:
        df = pd.read_csv(csv_path)
        df.drop_duplicates(subset="img_name")
        current_id = len(df)
    print("the current img id ", current_id)
    img_name_list = []
    best_shadow_list = []
    reject_reason_list = []
    for id_tmp, img_name in enumerate(img_names[current_id:]):
        id = id_tmp + current_id
        print("{} images have been process ({})) ".format(id, len(img_names)))
        img_name_list.append(img_name)
        img_path = os.path.join(shadow_mask_path, img_name)
        img = cv2.imread(img_path)
        cv2.namedWindow(str(id) + "_" + img_name, cv2.WINDOW_NORMAL)
        cv2.imshow(str(id) + "_" + img_name, img)
        key = cv2.waitKey(0)
        if key == ord('s'):
            best_shadow_list.append(1)
            reject_reason_list.append(0)
        else:
            best_shadow_list.append(0)
            # reject_reason = input("specify reject reason please: ")
            # reject_reason_list.append(reject_reason)

        cv2.destroyAllWindows()

        df_tmp = pd.DataFrame(data=zip(img_name_list, best_shadow_list), columns=['img_name', 'best_shadow'])
        df_final = pd.concat([df, df_tmp])
        df_final.drop_duplicates(subset="img_name")
        df_final.to_csv(csv_path,index=False)


if __name__ == '__main__':
    main()