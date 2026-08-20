import pytest
from unittest.mock import MagicMock, patch
from src.search.retriever import SearchRetriever

@pytest.fixture
def retriever():
    # Mock Qdrant and Embedder so tests run instantly without downloading models
    with patch('src.search.retriever.QdrantClient'), \
         patch('src.search.retriever.SentenceTransformer'):
        retriever = SearchRetriever()
        
        # Mock the embedder encoding function
        retriever._encode_text = MagicMock(return_value=[0.1] * 1536)
        
        yield retriever

def test_kis_query(retriever):
    """
    Test KIS with the user's sample query.
    Query: 'Đoạn clip cần tìm bắt đầu bằng cảnh 1 người đầu bếp đặt đang đặt món gỏi cuốn chay bày trên đĩa...'
    """
    query = "Đoạn clip cần tìm bắt đầu bằng cảnh 1 người đầu bếp đặt đang đặt món gỏi cuốn chay bày trên đĩa, nhân gồm rau xanh cuộn tròn và đậu hũ, gói trong bánh tráng màu vàng và tím. Đĩa được trang trí thêm lá xanh và hoa pansy tím-vàng, tạo cảm giác thanh mát và tinh tế."
    
    # Mock Qdrant returning a mock point
    mock_point = MagicMock()
    mock_point.score = 0.95
    mock_point.payload = {
        "video_id": "vid_001",
        "frame_index": 1200,
        "point_type": "keyframe",
        "image_path": "output/vid_001/keyframes/seg001/keyframe_00001200.jpg"
    }
    retriever.qdrant.search.return_value = [mock_point]
    
    results = retriever.kis_search(query, top_k=5)
    
    assert len(results) == 1
    assert results[0]["video_id"] == "vid_001"
    assert results[0]["frame_index"] == 1200
    assert results[0]["score"] == 0.95
    retriever._encode_text.assert_called_once_with(query)

def test_trake_query(retriever):
    """
    Test TRAKE with sequential moments.
    """
    queries = [
        "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.",
        "E2: Khoảnh khắc đầu tiên thấy miến măng tây đầu tiên tiếp xúc với dầu trong chảo.",
        "E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.",
        "E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn trên dĩa."
    ]
    
    # Mock Qdrant returning mock points for each query
    # E1 returns frame 100, E2 returns frame 200, etc. (Chronological)
    def mock_search(*args, **kwargs):
        query_vec = kwargs.get('query_vector')
        mock_pt = MagicMock()
        mock_pt.payload = {"video_id": "vid_cooking", "frame_index": 100} # Mock logic
        return [mock_pt]
        
    retriever.qdrant.search.side_effect = mock_search
    
    paths = retriever.trake_search(queries, top_k_per_query=5)
    
    assert retriever._encode_text.call_count == 4

def test_qa_query(retriever):
    """
    Test QA Context gathering.
    Question: 'Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã thuộc tỉnh Khánh Hòa. Hỏi xã này có tên là gì?'
    """
    question = "Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã thuộc tỉnh Khánh Hòa. Hỏi xã này có tên là gì?"
    
    # Mock Qdrant returning a segment caption
    mock_point = MagicMock()
    mock_point.score = 0.88
    mock_point.payload = {
        "video_id": "vid_fana",
        "point_type": "caption",
        "caption": "Video showing FANA club giving charity in a local commune. There is a banner that says 'Xã Suối Cát'."
    }
    retriever.qdrant.search.return_value = [mock_point]
    
    results = retriever.qa_search_context(question, top_k=3)
    
    assert len(results) == 1
    assert results[0]["video_id"] == "vid_fana"
    assert "Xã Suối Cát" in results[0]["text_caption"]
