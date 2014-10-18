//glm::vec2 Q0(min.x, float(j));
//glm::vec2 Q1(max.x, float(j));
//glm::vec2 u = Q1 - Q0;
//float s;
//float t;
//float minS = 1.0f, maxS = 0.0f;
//glm::vec2 v0(tri.p1 - tri.p0);
//glm::vec2 v1(tri.p2 - tri.p1);
//glm::vec2 v2(tri.p0 - tri.p2);
//
//glm::vec2 w;
//if(abs(u.x*v0.y - u.y*v0.x) > 1e-6) // not parallel
//{
//	w = Q0 - glm::vec2(tri.p0.x, tri.p0.y);
//	s = (v0.y*w.x - v0.x*w.y) / (v0.x*u.y - v0.y*u.x);
//	t = (u.x*w.y  - u.y*w.x ) / (u.x*v0.y - u.y*v0.x);
//	if(s > -1e-6 && s < 1+1e-6 && t > -1e-6 && t < 1+1e-6)
//	{
//		minS = glm::min(s, minS);
//		maxS = glm::max(s, maxS);
//	}
//}
//if(abs(u.x*v1.y - u.y*v1.x) > 1e-6) // not parallel
//{
//	w = Q0 - glm::vec2(tri.p1.x, tri.p1.y);
//	s = (v1.y*w.x - v1.x*w.y) / (v1.x*u.y - v1.y*u.x);
//	t = (u.x*w.y  - u.y*w.x ) / (u.x*v1.y - u.y*v1.x);
//	if(s > -1e-6 && s < 1+1e-6 && t > -1e-6 && t < 1+1e-6)
//	{
//		minS = glm::min(s, minS);
//		maxS = glm::max(s, maxS);
//	}
//}
//if(abs(u.x*v2.y - u.y*v2.x) > 1e-6) // not parallel
//{
//	w = Q0 - glm::vec2(tri.p2.x, tri.p2.y);
//	s = (v2.y*w.x - v2.x*w.y) / (v2.x*u.y - v2.y*u.x);
//	t = (u.x*w.y  - u.y*w.x ) / (u.x*v2.y - u.y*v2.x);
//	if(s > -1e-6 && s < 1+1e-6 && t > -1e-6 && t < 1+1e-6)
//	{
//		minS = glm::min(s, minS);
//		maxS = glm::max(s, maxS);
//	}
//}