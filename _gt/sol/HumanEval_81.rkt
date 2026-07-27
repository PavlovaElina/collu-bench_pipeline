#lang racket
(define (numerical_letter_grade grades)
  (map (lambda (g)
         (cond [(>= g 4.0) "A+"] [(> g 3.7) "A"] [(> g 3.3) "A-"]
               [(> g 3.0) "B+"] [(> g 2.7) "B"] [(> g 2.3) "B-"]
               [(> g 2.0) "C+"] [(> g 1.7) "C"] [(> g 1.3) "C-"]
               [(> g 1.0) "D+"] [(> g 0.7) "D"] [(> g 0.0) "D-"]
               [else "E"]))
       grades))
