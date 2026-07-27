#lang racket
(define (digits n)
  (define ds (filter odd? (map (lambda (c) (- (char->integer c) 48)) (string->list (number->string n)))))
  (if (null? ds) 0 (apply * ds)))
