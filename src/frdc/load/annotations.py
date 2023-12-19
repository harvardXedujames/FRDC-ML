import os
import warnings
from warnings import warn

from label_studio_sdk import Client
from label_studio_sdk.data_manager import Filters, Column, Type, Operator

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = os.environ["LABEL_STUDIO_API_KEY"]

client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
client.check_connection()


class Task(dict):
    def get_bounds_and_labels(self) -> tuple[list[tuple[int, int]], list[str]]:
        bounds = []
        labels = []
        for ann_ix, ann in enumerate(self["annotations"]):
            results = ann["result"]
            for r_ix, r in enumerate(results):
                r: dict

                # We flatten the value dict into the result dict
                v = r.pop("value")
                r = {**r, **v}

                # Points are in percentage, we need to convert them to pixels
                r["points"] = [
                    (
                        int(x * r["original_width"] / 100),
                        int(y * r["original_height"] / 100),
                    )
                    for x, y in r["points"]
                ]

                # Only take the first label as this is not a multi-label task
                r["label"] = r.pop("polygonlabels")[0]
                if not r["closed"]:
                    warnings.warn(
                        f"Label for {r['label']} @ {r['points']} not closed. "
                        f"Skipping"
                    )
                    continue

                bounds.append(r["points"])
                labels.append(r["label"])

        return bounds, labels


def get_task(
    file_name: str = "chestnut_nature_park/20201218/result.jpg",
    project_id: int = 1,
):
    proj = client.get_project(project_id)
    # Get the task that has the file name
    filter = Filters.create(
        Filters.AND,
        [
            Filters.item(
                # The GS path is in the image column, so we can just filter on that
                Column.data("image"),
                Operator.CONTAINS,
                Type.String,
                file_name,
            )
        ],
    )
    tasks = proj.get_tasks(filter)

    if len(tasks) > 1:
        warn(f"More than 1 task found for {file_name}, using the first one")
    elif len(tasks) == 0:
        raise ValueError(f"No task found for {file_name}")

    return Task(tasks[0])
