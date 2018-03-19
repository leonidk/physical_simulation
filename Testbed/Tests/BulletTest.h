/*
* Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BULLET_TEST_H
#define BULLET_TEST_H

#include <cmath>

class BulletTest : public Test
{
  public:
	BulletTest()
	{

		{
			b2BodyDef bd;
			bd.type = b2_dynamicBody;
			bd.position.Set(0.0f, 4.0f);

      b2PolygonShape shape;
			//b2CircleShape shape;
      float rad = 0.25;
      int n = 12;
      b2Vec2 points[n];
      for (int i = 0; i < n; i++) {
        points[i].x = rad * std::sin(2.0 * M_PI * (float(i) / n));
        points[i].y = rad * std::cos(2.0 * M_PI * (float(i) / n));
      }

      shape.Set(points, n);

			//shape.m_radius = 0.25f;
			//m_x = RandomFloat(-1.0f, 1.0f);
			m_x = 0.20352793f;
			bd.position.Set(m_x, 10.0f);
			bd.bullet = true;

      b2FixtureDef fd;

      fd.shape = &shape;
      fd.friction = 1.0;
      fd.restitution = 0.00;
      fd.density = 100.0f;


			m_bullet = m_world->CreateBody(&bd);
			//auto fix = m_bullet->CreateFixture(&shape, 100.0f);
      m_bullet->CreateFixture(&fd);
      //m_bullet->SetFixedRotation(true);
		}
	}

	void Launch()
	{
		printf("LAUNCH CALLED\n");
		//m_body->SetTransform(b2Vec2(0.0f, 4.0f), 0.0f);
		//m_body->SetLinearVelocity(b2Vec2_zero);
		//m_body->SetAngularVelocity(0.0f);

		extern int32 b2_gjkCalls, b2_gjkIters, b2_gjkMaxIters;
		extern int32 b2_toiCalls, b2_toiIters, b2_toiMaxIters;
		extern int32 b2_toiRootIters, b2_toiMaxRootIters;

		b2_gjkCalls = 0;
		b2_gjkIters = 0;
		b2_gjkMaxIters = 0;

		b2_toiCalls = 0;
		b2_toiIters = 0;
		b2_toiMaxIters = 0;
		b2_toiRootIters = 0;
		b2_toiMaxRootIters = 0;
	}

	void Setup(Settings *settings)
	{
    m_world->SetGravity(b2Vec2(0.0f, settings->gravity));
		m_bodies.resize(settings->bodies.size());
		b2BodyDef bd;

		b2PolygonShape box;
		for (int i = 0; i < settings->bodies.size(); i++) {
      b2FixtureDef fd;

      fd.shape = &box;
      fd.friction = settings->friction;
      fd.restitution = settings->rest;
      fd.density = 1.0f;

			bd.position.Set(0.0f, 0.0f);
			bd.angle = 0.0;
			box.SetAsBox(settings->sizes[i].x, settings->sizes[i].y);
			m_bodies[i] = m_world->CreateBody(&bd);
      m_bodies[i]->CreateFixture(&fd);
			//auto fix = m_bodies[i]->CreateFixture(&box, 1.0f);
      //a
			//fix->SetRestitution(0.75);

			m_bodies[i]->SetTransform(settings->bodies[i], settings->rotations[i]);
			m_bodies[i]->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
			m_bodies[i]->SetAngularVelocity(0.0f);
		}
		m_bullet->SetTransform(b2Vec2(-30.0f, 40.0f), 0.0f);
		//m_bullet->SetLinearVelocity(b2Vec2(2.2f, 0.0f));
		m_bullet->SetLinearVelocity(b2Vec2(3.0f, -1.0f));
		m_bullet->SetAngularVelocity(0.0f);
	}
	void Step(Settings *settings)
	{
		Test::Step(settings);
		auto p = m_bullet->GetPosition();
		auto v = m_bullet->GetLinearVelocity();

		settings->p1 = p;
		settings->v1 = v;
		if (settings->doGUI)
			printf("%f %f %f %f\n", p.x, p.y, v.x, v.y);
		extern int32 b2_gjkCalls, b2_gjkIters, b2_gjkMaxIters;
		extern int32 b2_toiCalls, b2_toiIters;
		extern int32 b2_toiRootIters, b2_toiMaxRootIters;
		if (settings->doGUI)
		{
			if (b2_gjkCalls > 0)
			{
				g_debugDraw.DrawString(5, m_textLine, "gjk calls = %d, ave gjk iters = %3.1f, max gjk iters = %d",
									   b2_gjkCalls, b2_gjkIters / float32(b2_gjkCalls), b2_gjkMaxIters);
				m_textLine += DRAW_STRING_NEW_LINE;
			}

			if (b2_toiCalls > 0)
			{
				g_debugDraw.DrawString(5, m_textLine, "toi calls = %d, ave toi iters = %3.1f, max toi iters = %d",
									   b2_toiCalls, b2_toiIters / float32(b2_toiCalls), b2_toiMaxRootIters);
				m_textLine += DRAW_STRING_NEW_LINE;

				g_debugDraw.DrawString(5, m_textLine, "ave toi root iters = %3.1f, max toi root iters = %d",
									   b2_toiRootIters / float32(b2_toiCalls), b2_toiMaxRootIters);
				m_textLine += DRAW_STRING_NEW_LINE;
			}
		}
	}

	static Test *Create()
	{
		return new BulletTest;
	}

	std::vector<b2Body*> m_bodies;
	int m_num=0;
	b2Body *m_bullet;
	float32 m_x;
};

#endif
