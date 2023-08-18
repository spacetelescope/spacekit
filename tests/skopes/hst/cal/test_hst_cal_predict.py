import os
from pytest import mark
from moto import mock_s3
from spacekit.skopes.hst.cal.predict import local_handler, lambda_handler
import boto3


EXPECTED_PREDS = {
    "asn": {
        "j8zs05020": {"memBin": 3, "memVal": 18.89, "clockTime": 76479},
        "ic0k06010": {"memBin": 2, "memVal": 8.55, "clockTime": 13904},
        "la8mffg5q": {"memBin": 0, "memVal": 0.8, "clockTime": 295},
        "oc3p011i0": {"memBin": 0, "memVal": 0.47, "clockTime": 55},
    }
}


def put_moto_s3_object(dataset, bucketname):
    client = boto3.client("s3", region_name="us-east-1")
    fpath = f"tests/data/hstcal/predict/{dataset}_MemModelFeatures.txt"
    obj = f"control/{dataset}/{dataset}_MemModelFeatures.txt"
    with open(f"{fpath}", "rb") as f:
        client.upload_fileobj(f, bucketname, obj)


@mark.hst
@mark.cal
@mark.predict
@mark.parametrize("pipeline", [("asn")])
@mock_s3
def test_local_predict_handler(hst_cal_predict_visits, pipeline):
    bucketname = "spacekit_bucket"
    s3 = boto3.resource("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucketname)
    dataset = hst_cal_predict_visits[pipeline][0]  # "j8zs05020"
    put_moto_s3_object(dataset, bucketname)
    pred = local_handler(dataset, bucket_name=bucketname)
    for k, v in pred.predictions.items():
        assert v == EXPECTED_PREDS[pipeline][dataset][k]


@mark.hst
@mark.cal
@mark.predict
@mark.parametrize("pipeline", [("asn")])
@mock_s3
def test_local_handler_multiple(hst_cal_predict_visits, pipeline):
    bucketname = "spacekit_bucket"
    s3 = boto3.resource("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucketname)
    datasets = hst_cal_predict_visits[
        pipeline
    ]  # ["j8zs05020", "ic0k06010", "la8mffg5q", "oc3p011i0"]
    for dataset in datasets:
        put_moto_s3_object(dataset, bucketname)
        pred = local_handler(dataset, bucket_name=bucketname)
        for k, v in pred.predictions.items():
            assert v == EXPECTED_PREDS[pipeline][dataset][k]


@mark.hst
@mark.cal
@mark.predict
@mark.parametrize("pipeline", [("asn")])
@mock_s3
def test_lambda_handler(hst_cal_predict_visits, pipeline):
    bucketname = "spacekit_bucket"
    s3 = boto3.resource("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucketname)
    dataset = hst_cal_predict_visits[pipeline][
        0
    ]  # ["j8zs05020", "ic0k06010", "la8mffg5q", "oc3p011i0"]
    put_moto_s3_object(dataset, bucketname)
    os.environ["MODEL_PATH"] = "models/hst_cal"
    key = f"control/{dataset}/{dataset}_MemModelFeatures.txt"
    event = dict(Dataset=dataset, Bucket=bucketname, Key=key)
    preds = lambda_handler(event, None)
    for k, v in preds.items():
        assert v == EXPECTED_PREDS[pipeline][dataset][k]
