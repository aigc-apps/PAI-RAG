from db.models.evaluation.evaluator_config import EvaluatorConfigCreate, EvaluatorConfigEntity, EvaluatorConfigRead


def test_create_entity():
    create_obj = EvaluatorConfigCreate()
    create_obj.name = 'foo name'
    create_obj.type = 'foo type'
    create_obj.model_id = 'foo model id'
    create_obj.case_sensitive = True
    create_obj.ignore_punctuation = True
    create_obj.extra_parameters = {'ek1':1, 'ek2':'ev2'}
    create_obj.sensitive_parameters = {'sk1': 'sv1', 'sk2': 'sv2'}

    entity_obj = EvaluatorConfigEntity.from_create_entity(dataset_id = 'foo ds id', evaluator_config_create = create_obj)
    assert entity_obj.dataset_id == 'foo ds id'
    assert entity_obj.name == create_obj.name
    assert entity_obj.type == create_obj.type
    assert entity_obj.model_id == create_obj.model_id
    assert entity_obj.case_sensitive == create_obj.case_sensitive
    assert entity_obj.ignore_punctuation == create_obj.ignore_punctuation
    assert entity_obj.extra_parameters == create_obj.extra_parameters
    assert entity_obj.sensitive_parameters != create_obj.sensitive_parameters
    assert 'sk1' in entity_obj.sensitive_parameters
    assert 'sk2' in entity_obj.sensitive_parameters
    assert len(entity_obj.sensitive_parameters) == 2
    print(f"eval config entity encrypted values 1:{entity_obj.sensitive_parameters['sk1']}")
    print(f"eval config entity encrypted values 2:{entity_obj.sensitive_parameters['sk2']}")
    assert len(entity_obj.sensitive_parameters['sk1']) > 3
    assert len(entity_obj.sensitive_parameters['sk2']) > 3


    read_obj = EvaluatorConfigRead.from_config_entity(entity_obj)
    assert read_obj.id == entity_obj.id
    assert read_obj.dataset_id == entity_obj.dataset_id
    assert read_obj.created_at == entity_obj.created_at
    assert read_obj.updated_at == entity_obj.updated_at
    assert read_obj.name == entity_obj.name
    assert read_obj.type == entity_obj.type
    assert read_obj.model_id == entity_obj.model_id
    assert read_obj.case_sensitive == entity_obj.case_sensitive
    assert read_obj.ignore_punctuation == entity_obj.ignore_punctuation
    assert read_obj.extra_parameters == entity_obj.extra_parameters
    assert read_obj.sensitive_parameters != entity_obj.sensitive_parameters
    assert 'sk1' in read_obj.sensitive_parameters
    assert 'sk2' in read_obj.sensitive_parameters
    assert len(read_obj.sensitive_parameters) == 2
    print(f"eval config read entity decrypted values 1:{read_obj.sensitive_parameters['sk1']}")
    print(f"eval config read entity decrypted values 2:{read_obj.sensitive_parameters['sk2']}")
    assert read_obj.sensitive_parameters['sk1'] == 'sv1'
    assert read_obj.sensitive_parameters['sk2'] == 'sv2'
