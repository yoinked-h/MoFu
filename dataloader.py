"""
Dataset/Data loader for MoFU
"""
from glob import glob
from pathlib import Path

def get_file_contents(file_path: str|Path) -> str:
    """
    Returns the content of a file
    If the file doesnt exist, raises a FileNotFoundError
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    with open(file_path, "r", encoding='utf-8') as _:
        return _.read()
class Dataset():
    """
    Dataset class to load data from a directory
    """
    def __init__(self, dataset_dir: str|Path = Path("./data/")):
        if isinstance(dataset_dir, str):
            dataset_dir = Path(dataset_dir)
        self.dataset_dir = dataset_dir
        # make sure that dataset_dir is an actual path
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} does not exist")
        self.files = []
        self.contents = {}
        self._prepare()
    def _prepare(self):
        for file in glob(str(self.dataset_dir / "**/*.txt"), recursive=True):
            filecontents = get_file_contents(file)
            if filecontents.startswith("##ignore"):
                continue
            filecontents = self._convert(filecontents)
            self.files.append(file)
            self.contents[file] = filecontents
    def get_tags_and_usage(self) -> dict:
        """
        returns a dict with the tags and times used in the dataset
        """
        tag_dict = {}
        for content in self.contents.values():
            tags = content.split(",")
            for tag in tags:
                if tag == "" or tag == " ":
                    continue 
                tag_dict[tag] = tag_dict.get(tag, 0) + 1
        return {k: v for k, v in sorted(tag_dict.items(), key=lambda item: item[1], reverse=True)}
            
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        return self.contents[self.files[idx]]
    def __iter__(self):
        for file in self.files:
            yield self.contents[file]
    def _convert(self, text):
        if "_" in text:
            text.replace(" ", ",")
            text.replace("_", " ")
        return text