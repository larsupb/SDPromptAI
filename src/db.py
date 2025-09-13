import logging
import re
import sqlite3
from typing import Set, List


def curate(prompt):
    # Remove leading/trailing whitespace and convert to lowercase
    prompt = prompt.strip().lower()
    # Remove line breaks and excessive spaces
    prompt = re.sub(r'\s+', ' ', prompt)
    return prompt


class DB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.db_path)
        if not self.check_table_exists("images"):
            cur = conn.cursor()
            cur.execute("""
                        CREATE TABLE images
                        (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            orgId           INTEGER,
                            url             TEXT UNIQUE,
                            hash            TEXT,
                            width           INTEGER,
                            height          INTEGER,
                            nsfw            BOOLEAN,
                            nsfwLevel       TEXT,
                            createdAt       TEXT,
                            postId          INTEGER,
                            username        TEXT,
                            seed            INTEGER,
                            model           TEXT,
                            steps           INTEGER,
                            sampler         TEXT,
                            cfgScale        REAL,
                            clipSkip        INTEGER,
                            positive_prompt TEXT,
                            negative_prompt TEXT
                        )
                        """)
            conn.commit()
        conn.close()

    def prune_images(self, min_length=64):
        # connect to db and delete images where positive_prompt is null or empty or below 64 characters
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM images WHERE positive_prompt IS NULL OR positive_prompt = '' OR LENGTH(positive_prompt) < " + str(min_length))
        deleted_rows = cur.rowcount
        conn.commit()
        conn.close()
        print(f"Pruned {deleted_rows} images from the database.")

    def insert_tags(self, tags):
        def prepare_params(tag):
            return (tag['id'], tag['url'], tag['hash'], tag['width'], tag['height'],
                    tag['nsfw'], tag['nsfwLevel'],
                    tag['createdAt'], tag['postId'], tag['username'],
                    tag.get('meta', {}).get('seed'),
                    tag.get('meta', {}).get('Model'),
                    tag.get('meta', {}).get('steps'),
                    tag.get('meta', {}).get('sampler'),
                    tag.get('meta', {}).get('cfgScale'),
                    tag.get('meta', {}).get('Clip skip'),
                    tag.get('meta', {}).get('prompt'),
                    tag.get('meta', {}).get('negativePrompt', None))

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for tag in tags:
            cur.execute("""
                        INSERT INTO images (orgId, url, hash, width, height, nsfw, nsfwLevel, createdAt, postId,
                                            username,
                                            seed, model, steps, sampler, cfgScale, clipSkip, positive_prompt,
                                            negative_prompt)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, prepare_params(tag))
        conn.commit()
        conn.close()
        print(f"Inserted {len(tags)} tags into the database.")

    def fetch_prompts(self, image_ids):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        prompts = []
        for image_id in image_ids:
            cur.execute(f"SELECT positive_prompt FROM images WHERE id={image_id}")
            row = cur.fetchone()
            if row:
                prompts.append(curate(row[0]))
            else:
                logging.warning("Could not find prompt for image id {}".format(image_id))
        conn.close()
        return prompts


    def fetch_all_prompts(self, prompt_column="positive_prompt", n_top=-1) -> (List[int], List[str]):
        if prompt_column not in ["positive_prompt", "interpreted_prompt"]:
            raise ValueError("prompt_column must be either 'positive_prompt' or 'interpreted_prompt'")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        prompt = f"SELECT Id, {prompt_column} FROM images WHERE interpreted_prompt IS NOT NULL"  # TODO Remove hack
        if n_top > 0:
            prompt = prompt + " LIMIT " + str(n_top)
        cur.execute(prompt)
        rows = cur.fetchall()

        conn.close()
        # Join all prompts into a single string

        # Create a list of prompts
        ids = []
        positive_prompts = []
        for row in rows:
            ids = ids + [row[0]]
            positive_prompts.append(curate(row[1]))

        return ids, positive_prompts

    def fetch_all_zero_rating_prompts(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, positive_prompt FROM images WHERE rating IS NULL")
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_image_ids_from_db(self) -> Set:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT orgId FROM images")
        rows = cur.fetchall()
        return {row[0] for row in rows}

    def get_image_prompts_from_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT positive_prompt FROM images")
        rows = cur.fetchall()
        return {row[0] for row in rows}

    def check_table_exists(self, table_name):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name = ?
                    """, (table_name,))
        return cur.fetchone() is not None

    def update_image_rating(self, image_id, rating_value):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE images SET rating=? WHERE id=?", (rating_value, image_id))
        conn.commit()
        conn.close()

    def fetch_random_prompts(self, prompt_count, prompt_column="interpreted_prompt"):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT {prompt_column} FROM images WHERE interpreted_prompt IS NOT NULL ORDER BY RANDOM() LIMIT {prompt_count}")
        rows = cur.fetchall()

        out = []
        for row in rows:
            out.append(curate(row[0]))

        conn.close()
        return out

    def fetch_all_uninterpreted_prompts(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, positive_prompt FROM images WHERE interpreted_prompt IS NULL ORDER BY RANDOM()")
        rows = cur.fetchall()
        conn.close()
        return rows

    def update_image_interpreted_prompt(self, image_id, interpreted_prompt):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE images SET interpreted_prompt=? WHERE id=?", (interpreted_prompt, image_id))
        conn.commit()
        conn.close()