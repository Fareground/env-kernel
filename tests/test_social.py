"""Tests for kernel/social.py"""
from unittest.mock import MagicMock
from fg_env_kernel.social import (
    ContentType, ContentVisibility,
    ContentItem, SocialGraph, ReputationSystem, ViralSpreadModel,
    Feed, SocialPlatformManager,
)


class TestContentItem:
    def test_basic(self):
        item = ContentItem(author_id="a1", text="Hello world")
        assert item.author_id == "a1"
        assert item.content_type == ContentType.POST
        assert item.id.startswith("content_")

    def test_total_engagement(self):
        item = ContentItem(
            reactions={"like": 5, "dislike": 2},
            share_count=3,
        )
        assert item.total_engagement() == 10

    def test_serialization(self):
        item = ContentItem(
            author_id="a1",
            text="Test post",
            content_type=ContentType.REPLY,
            parent_id="parent_1",
            visibility=ContentVisibility.FOLLOWERS_ONLY,
            tags=["news"],
            round_created=5,
            reactions={"like": 3},
            share_count=1,
        )
        item.reach.add("a1")
        item.reach.add("a2")
        d = item.to_dict()
        restored = ContentItem.from_dict(d)
        assert restored.author_id == "a1"
        assert restored.content_type == ContentType.REPLY
        assert restored.parent_id == "parent_1"
        assert restored.visibility == ContentVisibility.FOLLOWERS_ONLY
        assert "news" in restored.tags
        assert restored.reactions["like"] == 3
        assert "a1" in restored.reach


class TestSocialGraph:
    def test_follow(self):
        g = SocialGraph()
        g.follow("a", "b")
        assert g.is_following("a", "b")
        assert not g.is_following("b", "a")
        assert "b" in g.get_following("a")
        assert "a" in g.get_followers("b")

    def test_unfollow(self):
        g = SocialGraph()
        g.follow("a", "b")
        g.unfollow("a", "b")
        assert not g.is_following("a", "b")

    def test_friend(self):
        g = SocialGraph()
        g.friend("a", "b")
        assert "b" in g.get_friends("a")
        assert "a" in g.get_friends("b")

    def test_unfriend(self):
        g = SocialGraph()
        g.friend("a", "b")
        g.unfriend("a", "b")
        assert "b" not in g.get_friends("a")
        assert "a" not in g.get_friends("b")

    def test_block(self):
        g = SocialGraph()
        g.block("a", "b")
        assert g.is_blocked("a", "b")
        assert not g.is_blocked("b", "a")

    def test_mute(self):
        g = SocialGraph()
        g.mute("a", "b")
        assert "b" in g.get_muted("a")
        g.unmute("a", "b")
        assert "b" not in g.get_muted("a")

    def test_influence_score(self):
        g = SocialGraph()
        g.follow("b", "a")
        g.follow("c", "a")
        g.friend("a", "d")
        # a has 2 followers + 1 friend*0.5 = 2.5
        assert g.get_influence_score("a") == 2.5

    def test_echo_chamber_score(self):
        g = SocialGraph()
        # Simple case: a follows b and c, b and c follow each other
        g.follow("a", "b")
        g.follow("a", "c")
        g.follow("b", "c")
        g.follow("c", "b")
        score = g.get_echo_chamber_score("a")
        assert score > 0.0

    def test_echo_chamber_empty(self):
        g = SocialGraph()
        assert g.get_echo_chamber_score("nobody") == 0.0

    def test_remove_entity(self):
        g = SocialGraph()
        g.follow("a", "b")
        g.follow("b", "a")
        g.friend("a", "c")
        g.remove_entity("a")
        assert "a" not in g.get_followers("b")
        assert "a" not in g.get_friends("c")

    def test_serialization(self):
        g = SocialGraph()
        g.follow("a", "b")
        g.friend("c", "d")
        d = g.to_dict()
        restored = SocialGraph.from_dict(d)
        assert restored.is_following("a", "b")
        assert "d" in restored.get_friends("c")


class TestReputationSystem:
    def test_default_reputation(self):
        rep = ReputationSystem()
        assert rep.get_reputation("unknown") == 0.5

    def test_set_reputation(self):
        rep = ReputationSystem()
        rep.set_reputation("a", 0.8)
        assert rep.get_reputation("a") == 0.8

    def test_clamp(self):
        rep = ReputationSystem()
        rep.set_reputation("a", 1.5)
        assert rep.get_reputation("a") == 1.0
        rep.set_reputation("a", -0.5)
        assert rep.get_reputation("a") == 0.0

    def test_update_from_action(self):
        rep = ReputationSystem()
        rep.update_from_action("a", "help", success=True)
        assert rep.get_reputation("a") > 0.5
        rep.update_from_action("a", "steal", success=False)
        # Should still be above initial since success delta > failure delta
        assert rep.get_reputation("a") > 0.5

    def test_update_from_content(self):
        rep = ReputationSystem()
        content = ContentItem(author_id="a", reactions={"like": 10}, share_count=5)
        rep.update_from_content("a", content)
        assert rep.get_reputation("a") > 0.5

    def test_reputation_modifiers(self):
        rep = ReputationSystem()
        rep.set_reputation("a", 0.9)
        mods = rep.get_reputation_modifiers("a")
        assert mods["trust_bonus"] > 0
        assert mods["persuasion_modifier"] == 0.9

    def test_serialization(self):
        rep = ReputationSystem()
        rep.set_reputation("a", 0.7)
        d = rep.to_dict()
        restored = ReputationSystem.from_dict(d)
        assert restored.get_reputation("a") == 0.7


class TestViralSpreadModel:
    def test_simple_cascade(self):
        model = ViralSpreadModel(model_type="simple_cascade", spread_probability=1.0)
        graph = SocialGraph()
        graph.follow("b", "a")  # b follows a
        graph.follow("c", "a")
        content = ContentItem(author_id="a")
        already_reached = {"a"}
        new = model.calculate_spread(content, graph, already_reached)
        assert "b" in new
        assert "c" in new

    def test_simple_cascade_no_spread(self):
        model = ViralSpreadModel(model_type="simple_cascade", spread_probability=0.0)
        graph = SocialGraph()
        graph.follow("b", "a")
        new = model.calculate_spread(ContentItem(author_id="a"), graph, {"a"})
        assert len(new) == 0

    def test_threshold_model(self):
        model = ViralSpreadModel(model_type="threshold_model")
        graph = SocialGraph()
        graph.follow("target", "a")
        graph.follow("target", "b")
        # target follows a and b. If both are reached, threshold (30%) is met
        new = model.calculate_spread(ContentItem(), graph, {"a", "b"})
        assert "target" in new

    def test_persuasion(self):
        model = ViralSpreadModel(spread_probability=0.5)
        prob = model.apply_persuasion(source_reputation=1.0)
        assert prob > 0.5

    def test_serialization(self):
        model = ViralSpreadModel(model_type="threshold_model", spread_probability=0.5)
        d = model.to_dict()
        restored = ViralSpreadModel.from_dict(d)
        assert restored.model_type == "threshold_model"
        assert restored.spread_probability == 0.5


class TestFeed:
    def test_add_and_get(self):
        feed = Feed()
        item = ContentItem(text="Hello")
        feed.add_item(item)
        items = feed.get_feed(10)
        assert len(items) == 1
        assert items[0].text == "Hello"

    def test_max_items(self):
        feed = Feed(max_items=3)
        for i in range(5):
            feed.add_item(ContentItem(text=f"Post {i}"))
        assert feed.get_count() == 3

    def test_feed_order(self):
        feed = Feed()
        feed.add_item(ContentItem(text="First"))
        feed.add_item(ContentItem(text="Second"))
        items = feed.get_feed(2)
        assert items[0].text == "Second"  # Most recent first


class TestSocialPlatformManager:
    def test_create_content(self):
        mgr = SocialPlatformManager()
        item = mgr.create_content("a1", "Hello world", round_number=1)
        assert item.author_id == "a1"
        assert item.text == "Hello world"
        assert mgr.get_content(item.id) is item

    def test_react_to_content(self):
        mgr = SocialPlatformManager()
        item = mgr.create_content("a1", "Post")
        mgr.react_to_content("a2", item.id, "like")
        assert item.reactions["like"] == 1
        assert "a2" in item.reach

    def test_share_content(self):
        mgr = SocialPlatformManager()
        original = mgr.create_content("a1", "Original post")
        share = mgr.share_content("a2", original.id, round_number=2)
        assert share is not None
        assert share.content_type == ContentType.SHARE
        assert share.parent_id == original.id
        assert original.share_count == 1

    def test_share_nonexistent(self):
        mgr = SocialPlatformManager()
        result = mgr.share_content("a1", "nonexistent")
        assert result is None

    def test_get_feed(self):
        mgr = SocialPlatformManager()
        mgr.create_content("a1", "Post 1")
        mgr.create_content("a1", "Post 2")
        feed = mgr.get_feed("a1", limit=5)
        assert len(feed) == 2

    def test_get_trending(self):
        mgr = SocialPlatformManager()
        item1 = mgr.create_content("a1", "Viral post")
        item1.reactions["like"] = 100
        mgr.create_content("a2", "Normal post")
        trending = mgr.get_trending(limit=2)
        assert trending[0].text == "Viral post"

    def test_tick_spreads_content(self):
        mgr = SocialPlatformManager(
            viral_model=ViralSpreadModel(spread_probability=1.0),
        )
        mgr.social_graph.follow("b", "a")
        item = mgr.create_content("a", "Spreading post")
        mgr.tick(MagicMock(), round_number=1)
        # b should now have the item in their feed
        b_feed = mgr.get_feed("b")
        assert len(b_feed) == 1
        assert "b" in item.reach

    def test_tick_blocks_spread(self):
        mgr = SocialPlatformManager(
            viral_model=ViralSpreadModel(spread_probability=1.0),
        )
        mgr.social_graph.follow("b", "a")
        mgr.social_graph.block("b", "a")  # b blocks a
        item = mgr.create_content("a", "Blocked post")
        mgr.tick(MagicMock(), round_number=1)
        assert "b" not in item.reach

    def test_perception_data(self):
        mgr = SocialPlatformManager()
        mgr.social_graph.follow("b", "a")
        mgr.create_content("a", "Test post")
        data = mgr.get_perception_data("a")
        assert "social_feed" in data
        assert "follower_count" in data
        assert "reputation" in data

    def test_remove_entity(self):
        mgr = SocialPlatformManager()
        mgr.social_graph.follow("a", "b")
        mgr.create_content("a", "Post")
        mgr.remove_entity("a")
        assert "a" not in mgr.social_graph.get_following("a")

    def test_serialization(self):
        mgr = SocialPlatformManager()
        mgr.social_graph.follow("a", "b")
        item = mgr.create_content("a", "Test post", round_number=1)
        mgr.react_to_content("b", item.id, "like")

        d = mgr.to_dict()
        restored = SocialPlatformManager.from_dict(d)
        assert restored.social_graph.is_following("a", "b")
        content_ids = list(restored._content.keys())
        assert len(content_ids) == 1
        assert restored._content[content_ids[0]].text == "Test post"

    def test_empty_serialization(self):
        mgr = SocialPlatformManager()
        d = mgr.to_dict()
        restored = SocialPlatformManager.from_dict(d)
        assert len(restored._content) == 0

    def test_no_reputation(self):
        mgr = SocialPlatformManager(reputation_enabled=False)
        assert mgr.reputation is None
        data = mgr.get_perception_data("a")
        assert "reputation" not in data
