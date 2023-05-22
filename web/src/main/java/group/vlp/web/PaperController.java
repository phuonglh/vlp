package group.vlp.web;

import java.util.LinkedList;
import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class PaperController {
	static List<Paper> papers = new LinkedList<>();
	
	static {
		papers.add(new Journal(new String[]{"Phuong Le-Hong", "Erik Cambria"}, 
			"A Semantics-Aware Approach for Multilingual Natural Language Inference", 
			"Language Resources and Evaluation",
			2023, true, "Springer Nature", 57, 2, "611-649"));
		papers.add(new Journal(new String[]{"Chi Tho Luong", "Phuong Le-Hong", "Thi Oanh Tran"}, 
				"A rich task-oriented dialogue corpus in Vietnamese", 
				"Language Resources and Evaluation",
				2022, false, "Springer Nature", 0, 0, ""));
		papers.add(new Conference(new String[]{"Phuong Le-Hong", "et al."}, 
				"Multilingual Natural Language Understanding for the FPT.AI Conversational Platform",
				"Proceedings of the 14th IEEE International Conference on Knowledge and Systems Engineering",
				2022, true, "IEEE", "Nha Trang, Vietnam"));
		
	}
	
	@GetMapping(path = "/papers")
	public List<Paper> getPapers() {
		return papers;
	}
}
