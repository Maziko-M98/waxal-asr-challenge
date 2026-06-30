from io import StringIO

from waxal_asr.zindi_csv import read_zindi_csv


def test_read_zindi_csv_handles_escaped_quotes():
    csv_text = 'id,transcription,language,original_split\nx1,"hello \\"waxal\\"",lug,train\n'
    table = read_zindi_csv(StringIO(csv_text))
    assert table.loc[0, "transcription"] == 'hello "waxal"'
