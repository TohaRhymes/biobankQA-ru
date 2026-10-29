import unittest
import importlib, os
p=importlib.import_module(os.environ.get("SECTION_TEST_MODULE", "benchmark.scripts.parse_materials_v2"))
class Typography(unittest.TestCase):
 def test_large_bold_heading(self):self.assertTrue(p._is_typographic_heading('Материалы и методы',13,10,.95))
 def test_body_not_heading(self):self.assertFalse(p._is_typographic_heading('Образцы были переданы в лабораторию.',10,10,0))
 def test_caption_not_heading(self):self.assertFalse(p._is_typographic_heading('Таблица 2. Список образцов',13,10,1))
 def test_running_header_not_heading(self):self.assertFalse(p._is_typographic_heading('Кардиоваскулярная терапия и профилактика',13,10,1))
 def test_toc_not_heading(self):self.assertFalse(p._is_typographic_heading('Материалы и методы ........ 25',13,10,1))
 def test_body_font_bold_subheading(self):self.assertTrue(p._is_typographic_heading('Результаты исследования',10,10,1))
 def test_key_message_box_not_section(self):self.assertFalse(p._is_typographic_heading('Что известно о предмете исследования?',10,10,1))
 def test_rotated_garbage_not_section(self):self.assertFalse(p._is_typographic_heading('AIBKFN 696sr -изомог GG',13,10,1))
 def test_references_heading_not_carried(self):self.assertFalse(p._is_typographic_heading('Литература/References',10,10,1))
 def test_body_font_regular_not_heading(self):self.assertFalse(p._is_typographic_heading('Материалы и методы',10,10,0))
 def test_column_cut_fragment_is_not_promoted(self):
  from types import SimpleNamespace
  page=SimpleNamespace(width=200,height=100,chars=[dict(text='Заголовок',size=10,fontname='Font-Bold',top=10,x0=0,x1=40),dict(text='продолжение',size=10,fontname='Font-Bold',top=10,x0=44,x1=100)])
  self.assertEqual(p._typographic_heading_lines(page,[dict(text='Заголовок',top=10,x0=0,x1=40)]),[])
 def test_standalone_heading_has_font_evidence(self):
  from types import SimpleNamespace
  page=SimpleNamespace(width=200,height=100,chars=[dict(text='Введение',size=10,fontname='Font-Bold',top=10,x0=0,x1=40)])
  self.assertEqual(p._typographic_heading_lines(page,[dict(text='Введение',top=10,x0=0,x1=40)]),[('Введение',.1)])
 def test_hint_uses_original_text_unchanged(self):
  e=p.Elem(1,'NarrativeText','Введение\nДлинный текст тела.',0,100,0,.5,section_hint='Введение');self.assertEqual(p.detect_section_for_page([e]),'Введение');self.assertEqual(e.text,'Введение\nДлинный текст тела.')
if __name__=='__main__':unittest.main()
