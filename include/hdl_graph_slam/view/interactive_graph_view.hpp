#ifndef HDL_GRAPH_SLAM_INTERACTIVE_GRAPH_VIEW_HPP
#define HDL_GRAPH_SLAM_INTERACTIVE_GRAPH_VIEW_HPP

#include <mutex>
#include <unordered_map>
#include <glk/glsl_shader.hpp>

#include <hdl_graph_slam/interactive_graph.hpp>

#include <hdl_graph_slam/view/edge_view.hpp>
#include <hdl_graph_slam/view/vertex_view.hpp>
#include <hdl_graph_slam/view/keyframe_view.hpp>
#include <hdl_graph_slam/view/line_buffer.hpp>
#include <hdl_graph_slam/view/drawable_object.hpp>

namespace hdl_graph_slam {

class InteractiveGraphView : public InteractiveGraph {
public:
  InteractiveGraphView() { }
  virtual ~InteractiveGraphView() override {}

  void init_gl() { line_buffer.reset(new LineBuffer()); }

  void update_view() {
    bool keyframe_inserted = false;
    for (const auto& key_item : keyframes) {
      auto& keyframe = key_item.second;
      auto found = keyframes_view_map.find(keyframe);
      if (found == keyframes_view_map.end()) {
        keyframe_inserted = true;
        keyframes_view.push_back(std::make_shared<KeyFrameView>(keyframe));
        keyframes_view_map[keyframe] = keyframes_view.back();

        vertices_view.push_back(keyframes_view.back());
        vertices_view_map[keyframe->id()] = keyframes_view.back();

        drawables.push_back(keyframes_view.back());
      }
    }

    if (keyframe_inserted) {
      std::sort(keyframes_view.begin(), keyframes_view.end(), [=](const KeyFrameView::Ptr& lhs, const KeyFrameView::Ptr& rhs) { return lhs->lock()->id() < rhs->lock()->id(); });
    }

    for (const auto& vertex : graph->vertices()) {
      auto found = vertices_view_map.find(vertex.second->id());
      if (found != vertices_view_map.end()) {
        continue;
      }

      auto vertex_view = VertexView::create(vertex.second);
      if (vertex_view) {
        vertices_view.push_back(vertex_view);
        vertices_view_map[vertex.second->id()] = vertex_view;

        drawables.push_back(vertex_view);
      }
    }

    for (const auto& edge : graph->edges()) {
      auto found = edges_view_map.find(edge);
      if (found != edges_view_map.end()) {
        continue;
      }

      auto edge_view = EdgeView::create(edge, *line_buffer);
      if (edge_view) {
        edges_view.push_back(edge_view);
        edges_view_map[edge] = edge_view;

        drawables.push_back(edge_view);
      }
    }
  }

  void draw(const DrawFlags& flags, glk::GLSLShader& shader) {
    update_view();
    line_buffer->clear();

    for (auto& drawable : drawables) {
      if (drawable->available()) {
        drawable->draw(flags, shader);
      }
    }

    line_buffer->draw(shader);
  }

  void delete_edge(EdgeView::Ptr edge) {
    std::cout << "delete edge " << edge->id() << std::endl;

    auto found = std::find(edges_view.begin(), edges_view.end(), edge);
    while(found != edges_view.end()) {
      edges_view.erase(found);
      found = std::find(edges_view.begin(), edges_view.end(), edge);
    }

    auto found2 = std::find(drawables.begin(), drawables.end(), edge);
    while(found2 != drawables.end()) {
      drawables.erase(found2);
      found2 = std::find(drawables.begin(), drawables.end(), edge);
    }

    for(auto itr = edges_view_map.begin(); itr != edges_view_map.end(); itr++) {
      if(itr->second == edge) {
        edges_view_map.erase(itr);
        break;
      }
    }

    graph->removeEdge(edge->edge);
  }
  //Rewrite this code
  std::vector<EdgeView::Ptr> find_connected_edges(VertexView::Ptr vertex)
  {
    std::vector<EdgeView::Ptr> edges_which_need_to_delete;
    for (const auto& edge : graph->vertices()[vertex->id()]->edges()) {
      auto edge_view = std::find_if(edges_view.begin(), edges_view.end(),
                                    [edge](EdgeView::Ptr& edge_view) { return edge->id() == edge_view->id(); });
      if (edge_view != edges_view.end()) {
        edges_which_need_to_delete.push_back(*edge_view);
      }
    }
    return edges_which_need_to_delete;
  }

  //Rewrite this code
  void delete_vertex(VertexView::Ptr vertex)
  {
    if (anchor_node_id() == vertex->id()) {
      return;
    } 

    std::cout << "delete vertex " << vertex->id() << std::endl;
    auto edges_to_be_deleted = find_connected_edges(vertex);
    for (auto& edge : edges_to_be_deleted) {
      delete_edge(edge);
    }


    // auto vertex_view = std::find(vertices_view.begin(), vertices_view.end(), vertex);
    // if (vertex_view != vertices_view.end()) {
    //   vertices_view.erase(vertex_view);
    // }

    // auto drawable = std::find(drawables.begin(), drawables.end(), vertex);
    // if (drawable != drawables.end()) {
    //   drawables.erase(drawable);
    // }

    // for (auto itr = vertices_view_map.begin(); itr != vertices_view_map.end(); itr++) {
    //   if (itr->second == vertex) {
    //     vertices_view_map.erase(itr);
    //     break;
    //   }
    // }

    // keyframes.erase(vertex->id());

    // if (anchor_node_id() == vertex->id()) {
    //   reset_anchor_node();
    // }

    // if (floor_node_id() == vertex->id()) {
    //   reset_floor_node();
    // }
    graph->removeVertex(graph->vertices()[vertex->id()]);

    keyframes_view.clear();
    keyframes_view_map.clear();
    vertices_view.clear();
    vertices_view_map.clear();
    edges_view.clear();
    edges_view_map.clear();
    drawables.clear();
    update_view();

  }

public:
  std::unique_ptr<LineBuffer> line_buffer;

  std::vector<KeyFrameView::Ptr> keyframes_view;
  std::unordered_map<InteractiveKeyFrame::Ptr, KeyFrameView::Ptr> keyframes_view_map;

  std::vector<VertexView::Ptr> vertices_view;
  std::unordered_map<long, VertexView::Ptr> vertices_view_map;

  std::vector<EdgeView::Ptr> edges_view;
  std::unordered_map<g2o::HyperGraph::Edge*, EdgeView::Ptr> edges_view_map;

  std::vector<DrawableObject::Ptr> drawables;
};

}  // namespace hdl_graph_slam

#endif