from pathlib import Path

from codesage.ingest import Chunk, chunk_file, iter_source_files, ingest_repo


def test_chunk_file_splits_by_line_window(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 91)))  # 90 lines

    chunks = chunk_file(f, lines_per_chunk=40)

    assert len(chunks) == 3
    assert chunks[0].line_start == 1 and chunks[0].line_end == 40
    assert chunks[1].line_start == 41 and chunks[1].line_end == 80
    assert chunks[2].line_start == 81 and chunks[2].line_end == 90
    assert chunks[0].text.startswith("line1\n")


def test_chunk_file_skips_blank_only_windows(tmp_path: Path):
    f = tmp_path / "blank.py"
    f.write_text("\n" * 45)  # all blank lines

    chunks = chunk_file(f, lines_per_chunk=40)

    assert chunks == []


def test_iter_source_files_skips_dot_git_and_non_text(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("ignored")

    found = {p.name for p in iter_source_files(tmp_path)}

    assert found == {"a.py"}


def test_ingest_repo_embeds_every_chunk(tmp_path: Path):
    (tmp_path / "a.py").write_text("\n".join(f"line{i}" for i in range(1, 5)))

    class FakeLLM:
        def embed(self, text, output_dimensionality=768):
            return [float(len(text))]

    chunks = ingest_repo(tmp_path, FakeLLM())

    assert len(chunks) == 1
    assert chunks[0].vector == [float(len(chunks[0].text))]
    assert isinstance(chunks[0], Chunk)
